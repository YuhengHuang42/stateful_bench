from enum import Enum
import os
import json
from loguru import logger


from Sgenerator.utils import write_jsonl
GENERATOR_PROMPT = '''You are given a block of sequential API calls produced by a fuzzing engine, along with an optional initialization block. Your task is to translate these API calls into precise natural language descriptions that could be used as programming instructions for a developer.

Requirements:
	1.	The description must be unambiguous and accurately capture the semantics of the program.
	2.	The output should closely resemble how a programmer would describe a task in a typical software development scenario (e.g., feature request, ticket description, or implementation notes).
	3.	Instead of word-by-word translation, write the description in a humanlike and scenario-driven style, as if you are specifying a real-world feature or task.
	4.	The initialization block gives contextual information. While the values might differ, the variable's names should be fixed. You should provide variable names and explain them in the task descriptions. They are the parameters to the program. 
	5.	When describing the task requirements, include as few constants as possible. 
	6.	Do not include code in the output — only natural language.
    7.  Since an API documentation will be provided later along with the natural language descriptions, it is OK to omit some details regarding API formal specifications -- only natural language descriptions.

{application_description}

Program:
```
# ===== Init Block Begin =====
{init_block}
# ===== Init Block End =====

{program}
```

In your description, first define user variable/constant name (such as user_variable_0) in the init block with necessary descriptions of their types and structures (e.g., dict with what key), but do not directly show their values since they may change with respect to different test cases.
Provide necessary instructions when these variable/constant names are referred to. ONLY refer to names, not values. 
All user-provided content should be used (e.g., when the variable value matches user_constant_0) if they are referred in the program.
If there is RESULT variable, specify it as its vaule will be checked using test cases.

Note: The generated description will be used to evaluate LLMs' ability to understand natural language and generate correct stateful code.
If you can make the description unambiguous, you should allow the LLM to infer implicit states rather than explicitly listing every detail.

Response Format:
<User Variable Definition>
user_variable_0 is a xxx, ...
<Task Instructions>
Implement ...
'''

GENERATOR_IMPROVE_PROMPT = ''' An evaluator agent has checked the descriptions and offered some advices. Please improve your description:
{evaluator_output}
'''

EVALUATOR_PROMPT = '''
You are a checker agent tasked with evaluating the quality of a natural language description that was generated from a sequence of API calls. Your job is to determine whether the description meets two key criteria:

1.  Fidelity to the Program Logic
	a. Does the natural language description accurately reflect all steps of the given API call sequence?
	b. Would a developer be able to reconstruct the original program logic (or a functionally equivalent one) based solely on the description?
	c. Are any steps missing, incorrectly described, or added?
	d. Is there any ambiguity? For example, sometimes "update" can refer to either "local update" or "remote update through APIs." 
2.  Human-likeness and Fluency
	a. Does the description sound natural and fluent, as if it were written by a human programmer?
    b. Does the description present the task in a plausible use-case or user scenario?
	c. Is the tone and phrasing consistent with how developers describe implementation tasks?
3.  Redundancy
	a. Is the description redundant? For example, if the description is too verbose or if it contains unnecessary details that could be inferred from other parts of the description.

Your output should be one of the following:
	•	If the description satisfies all the criteria: <OK>
	•	If any issues are found, provide a brief but specific diagnosis along with suggestions for improvement. Suggestions can be general (e.g., tone, structure) or targeted at a specific part of the description. Keep your feedback concise and actionable.
    •	If you find it is impossible to generate a description that satisfies all the criteria, output <IMPOSSIBLE>
Below is the program:
```
# ===== Init Block Begin =====
{init_block}
# ===== Init Block End =====

{program}
```

Below is the description:
# ===== Description Begin =====
{description}
# ===== Description End =====
'''

EVALUATOR_FURTHER_PROMPT = '''
Here are the updated descriptions according to your suggestions:
{description}

If the description satisfies all the criteria mentioned above, generate: <OK>
Otherwise, output a short diagnosis and suggestions for improvement. Be concise but specific.
'''

class AgentStatus(Enum):
    BEGIN = 0
    CONTINUE = 1
    END = 2

# ===== OPENAI API Related Functions=====
def generate_jsonl_for_openai(request_id_list, 
                              message_list, 
                              output_path,
                              max_tokens=None, 
                              model_type="gpt-4.1", 
                              url="/v1/chat/completions"):
    """
    Prepare input batch data for OPENAI API
    Args:
        request_id_list: used to index the message_list.
        message_list: list of messages to be sent to the API
        output_path: output file path
        max_tokens: maximum tokens to generate
        model_type: model type of OPENAI API
        url: API endpoint
    """
    assert len(request_id_list) == len(message_list)
    data = []
    requestid_to_message = dict(zip(request_id_list, message_list))
    for idx, item in enumerate(request_id_list):
        request_id = item
        message = message_list[idx]
        body = {"model": model_type, 
         "messages": message
        }
        if max_tokens is not None:
            body["max_tokens"] = max_tokens
        per_request = {
            "custom_id": request_id,
            "method": "POST",
            "url": url,
            "body": body
        }
        data.append(per_request)
    
    write_jsonl(output_path, data)
    
    return data, requestid_to_message

def submit_batch_request_openai(client, 
                                input_file_path, 
                                url="/v1/chat/completions", 
                                completion_window="24h", 
                                description="code analysis"):
    """
    Submit the batch task to OPENAI API
    Args:
        client: OPENAI API client (client = openai.OpenAI(api_key=api_key))
        input_file_path: input file path
        url: API endpoint
        completion_window: completion window
        description: description of the submitted task
    """
    batch_input_file = client.files.create(
        file=open(input_file_path, "rb"),
        purpose="batch"
    )
    batch_input_file_id = batch_input_file.id
    batch_submit_info = client.batches.create(
        input_file_id=batch_input_file_id,
        endpoint=url,
        completion_window=completion_window,
        metadata={
        "description": description
        }
    )
    batch_submit_info_id = batch_submit_info.id
    batch_result_info = client.batches.retrieve(batch_submit_info_id)
    
    return (batch_submit_info, batch_result_info)

class Generator():
    def __init__(self, application_description,
                 max_iterations,
                 generator_prompt=GENERATOR_PROMPT, 
                 improvement_prompt=GENERATOR_IMPROVE_PROMPT):
        self.application_description = application_description
        self.generator_prompt = generator_prompt
        self.improvement_prompt = improvement_prompt
        self.status = AgentStatus.BEGIN
        self.request_counter = 0
        self.max_iterations = max_iterations
        self.message = [{
            "role": "system",
            "content": None,
        }]
        
    def get_generate_prompt(self, init_block, program):
        return self.generator_prompt.format(
            application_description=self.application_description,
            init_block=init_block,
            program=program
        )
    
    def get_improve_prompt(self, evaluator_output):
        return self.improvement_prompt.format(
            evaluator_output=evaluator_output,
        )
    
    def analyze_output(self, return_string):
        # Split response into variable definitions and task instructions
        sections = return_string.split("<User Variable Definition>")
        if len(sections) < 2:
            raise ValueError("Invalid response format - missing User Variable Definition")
            
        var_section, remaining = sections[1].split("<Task Instructions>")
        
        init_info = var_section.strip()
        description = remaining.strip()
        
        return {
            "init_info": init_info,
            "description": description
        }
    
    def get_all_generated_description(self):
        assert self.message[-1]["role"] == "assistant"
        #content = self.analyze_output(self.message[-1]["content"])
        return self.message[-1]["content"]
    
    def interact(self, info_dict):
        if self.status == AgentStatus.BEGIN:
            assert "init_block" in info_dict
            assert "program" in info_dict
            self.message[0]["content"] = self.get_generate_prompt(info_dict["init_block"], info_dict["program"])
            self.request_counter += 1
            return True, self.message
        elif self.status == AgentStatus.CONTINUE:
            assert "evaluator_output" in info_dict
            self.message.append({
                "role": "user",
                "content": self.get_improve_prompt(info_dict["evaluator_output"])
            })
            self.request_counter += 1
            return True, self.message
        elif self.status == AgentStatus.END:
            return False, None
    
    def record_response(self, message):
        if self.status == AgentStatus.BEGIN:
            self.status = AgentStatus.CONTINUE
        self.message.append(message)
        if self.request_counter >= self.max_iterations:
            self.status = AgentStatus.END
        return self.status
    
    
class Evaluator():
    def __init__(self, application_description,
                 max_iterations, 
                 evaluator_prompt=EVALUATOR_PROMPT, 
                 further_prompt=EVALUATOR_FURTHER_PROMPT):
        self.application_description = application_description
        self.evaluator_prompt = evaluator_prompt
        self.further_prompt = further_prompt
        self.status = AgentStatus.BEGIN
        self.request_counter = 0
        self.max_iterations = max_iterations
        # Even when using previous_response_id, all previous input tokens for responses in the chain are billed as input tokens in the API.
        self.message = [{
            "role": "system",
            "content": None,
        }]
        
    def get_evaluate_prompt(self, init_block, program, description):
        return self.evaluator_prompt.format(
            init_block=init_block,
            program=program,
            description=description
        )
    
    def get_further_prompt(self, description):
        return self.further_prompt.format(
            description=description
        )
    
    def whether_continue(self, return_string):
        if "<OK>" in return_string or "<IMPOSSIBLE>" in return_string:
            return False
        else:
            return True
    
    def interact(self, info_dict):
        if self.status == AgentStatus.BEGIN:
            assert "init_block" in info_dict
            assert "program" in info_dict
            assert "description" in info_dict
            self.status = AgentStatus.CONTINUE
            self.message[0]["content"] = self.get_evaluate_prompt(info_dict["init_block"], info_dict["program"], info_dict["description"])
            self.request_counter += 1
            return True, self.message
        elif self.status == AgentStatus.CONTINUE:
            assert "description" in info_dict
            user_prompt = self.get_further_prompt(info_dict["description"])
            self.message.append({
                "role": "user",
                "content": user_prompt
            })
            self.request_counter += 1
            return True, self.message
        elif self.status == AgentStatus.END:
            return False, None
    
    def record_response(self, message):
        self.message.append(message)
        flag = self.whether_continue(message["content"])
        if flag:
            self.status = AgentStatus.CONTINUE
        else:
            self.status = AgentStatus.END
        if self.request_counter >= self.max_iterations:
            self.status = AgentStatus.END
        return flag
    
    def get_evaluator_output(self):
        assert self.message[-1]["role"] == "assistant"
        evaluator_output = {"evaluator_output": self.message[-1]["content"]}
        return evaluator_output

class MultiAgent():
    def __init__(self,
                 program_info,
                 max_iterations,
                 application_description,
                 openai_client,
                 wait_time=10,
                 generator_prompt=GENERATOR_PROMPT,
                 evaluator_prompt=EVALUATOR_PROMPT,
                 max_tokens=None,
                 model_type="gpt-4.1",
                 url="/v1/chat/completions"):
        '''
        Args:
            program_info: list of program info
                Each item is a dict with "init_block" and "program"
            max_iterations: maximum iterations
            application_description: application description
            openai_client: openai client
            wait_time: wait time
        '''
        self.agent_book = self.prepare_agent(program_info, max_iterations, application_description, generator_prompt, evaluator_prompt)
        self.max_iterations = max_iterations
        self.message_recorder = dict()
        self.max_tokens = max_tokens
        self.model_type = model_type
        self.url = url
        self.client = openai_client
        self.wait_time = wait_time
        
    def wait_for_batch_completion(self, batch_id):
        """Poll batch status until completion"""
        import time
        while True:
            batch_status = self.client.batches.retrieve(batch_id)
            if batch_status.status in ['completed', 'failed', 'cancelled']:
                return
            time.sleep(self.wait_time)  # Check every 10 seconds
            
    def prepare_agent(
        self,
        program_info,
        max_iterations,
        application_description,
        generator_prompt=GENERATOR_PROMPT,
        evaluator_prompt=EVALUATOR_PROMPT):
        agent_book = dict()
        for idx, item in enumerate(program_info):
            agent_book[idx] = {
                    "generator": Generator(application_description, max_iterations, generator_prompt),
                    "evaluator": Evaluator(application_description, max_iterations, evaluator_prompt)
            }
        return agent_book

    def wrap_multi_agent_message(
        self,
        program_info,
        agent_book,
        turn
        ):
        '''
        Perform multi-agent translation.
        Args:
            program_info: list of program info
            agent_book: agent book returned by prepare_agent
            turn: Generator -> Evaluator -> Generator -> ...
        '''
        message_list = []
        assert turn in ["Generator", "Evaluator"]
        for idx, item in enumerate(program_info):
            init_block = item["init_block"]
            init_load_str = item["init_load_str"]
            if init_load_str is not None:
                init_block = init_load_str + "\n" + init_block
            
            program = item["program"]
            generator = agent_book[idx]["generator"]
            evaluator = agent_book[idx]["evaluator"]
            if generator.status == AgentStatus.BEGIN:
                # Sequence begin with the generator
                generator_info = {
                    "init_block": init_block,
                    "program": program
                }
                flag, generator_output = generator.interact(generator_info)
                message_list.append(("generator_{}".format(idx), generator_output))
            else:
                if generator.status == AgentStatus.CONTINUE:
                    if evaluator.status == AgentStatus.BEGIN:
                        # The first time the evaluator is called, it needs the init_block and program information.
                        all_generated_description = generator.get_all_generated_description()
                        evaluator_info = {
                            "init_block": init_block,
                            "program": program,
                            "description": all_generated_description
                        }
                        flag, evaluator_output = evaluator.interact(evaluator_info)
                        message_list.append(("evaluator_{}".format(idx), evaluator_output))
                    elif evaluator.status == AgentStatus.CONTINUE:
                        if turn == "Evaluator":
                            all_generated_description = generator.get_all_generated_description()
                            evaluator_info = {
                                "description": all_generated_description
                            }
                            flag, evaluator_output = evaluator.interact(evaluator_info)
                            message_list.append(("evaluator_{}".format(idx), evaluator_output))
                        elif turn == "Generator":
                            evaluator_output = evaluator.get_evaluator_output()
                            flag, generator_output = generator.interact(evaluator_output)
                            message_list.append(("generator_{}".format(idx), generator_output))
                    else:
                        generator.status = AgentStatus.END
                        continue
                else:
                    evaluator.status = AgentStatus.END
                    continue
        return message_list

    def collect_multi_agent_message(
        self,
        openai_message):
        '''
        '''
        message_list = []
        for i in openai_message.iter_lines():
            message_list.append(json.loads(i))
        for item in message_list:
            custom_id = item["custom_id"]
            item_id = int(custom_id.split("_")[1])
            is_generator = custom_id.split("_")[0] == "generator"
            content = item["response"]["body"]["choices"][0]["message"]["content"]
            role = item["response"]["body"]["choices"][0]["message"]["role"]
            response = {
                "content": content,
                "role": role
            }
            if is_generator:
                generator = self.agent_book[item_id]["generator"]
                generator.record_response(response)
            else:
                evaluator = self.agent_book[item_id]["evaluator"]
                evaluator.record_response(response)
        
    
    def interact_loop(self, program_info, output_dir):
        def prepare_output_jsonl(rml, round_info):
            request_id_list = [item[0] for item in rml]
            message_list = [item[1] for item in rml]
            output_path = os.path.join(output_dir, "round_{}_openai_input.jsonl".format(round_info))
            data, requestid_to_message = generate_jsonl_for_openai(request_id_list, message_list, output_path, self.max_tokens, self.model_type, self.url)
            return data, requestid_to_message, output_path
        
        request_message_list = self.wrap_multi_agent_message(program_info, self.agent_book, "Generator")
        _, _, output_path = prepare_output_jsonl(request_message_list, "{}_generator".format(0))
        batch_submit_info, batch_result_info = submit_batch_request_openai(self.client, output_path, self.url, description="translation_{}_generator".format(0))
        
        logger.info("Waiting for batch completion at round {}".format(0))
        self.wait_for_batch_completion(batch_submit_info.id)
        batch_result_info = self.client.batches.retrieve(batch_submit_info.id)
        file_response = self.client.files.content(batch_result_info.output_file_id)
        self.collect_multi_agent_message(file_response)
        self.message_recorder["{}_generator".format(0)] = {
            "file_response": file_response,
            "batch_submit_info": batch_submit_info,
            "batch_result_info": batch_result_info
        }
        
        
        for round in range(1, self.max_iterations):
            # Evaluator
            request_message_list = self.wrap_multi_agent_message(program_info, self.agent_book, "Evaluator")
            if len(request_message_list) == 0:
                break
            _, _, output_path = prepare_output_jsonl(request_message_list, "{}_evaluator".format(round))    
            batch_submit_info, batch_result_info = submit_batch_request_openai(self.client, output_path, self.url, description="translation_{}_evaluator".format(round))
            logger.info("Waiting for batch completion at round {} for evaluator".format(round))
            self.wait_for_batch_completion(batch_submit_info.id)
            batch_result_info = self.client.batches.retrieve(batch_submit_info.id)
            file_response = self.client.files.content(batch_result_info.output_file_id)
            self.collect_multi_agent_message(file_response)
            self.message_recorder["{}_evaluator".format(round)] = {
                "file_response": file_response,
                "batch_submit_info": batch_submit_info,
                "batch_result_info": batch_result_info
            }
            
            # Generator 
            request_message_list = self.wrap_multi_agent_message(program_info, self.agent_book, "Generator")
            if len(request_message_list) == 0:
                break
            _, _, output_path = prepare_output_jsonl(request_message_list, "{}_generator".format(round))
            batch_submit_info, batch_result_info = submit_batch_request_openai(self.client, output_path, self.url, description="translation_{}_generator".format(round))
            logger.info("Waiting for batch completion at round {} for generator".format(round))
            self.wait_for_batch_completion(batch_submit_info.id)
            batch_result_info = self.client.batches.retrieve(batch_submit_info.id)
            file_response = self.client.files.content(batch_result_info.output_file_id)
            self.collect_multi_agent_message(file_response)
            self.message_recorder["{}_generator".format(round)] = {
                "file_response": file_response,
                "batch_submit_info": batch_submit_info,
                "batch_result_info": batch_result_info
            }
        
        return self.agent_book

    def save_agent_data(self):
        """Save agent states including message history and last evaluator output"""
        saved_data = {}
        for idx, agent_pair in self.agent_book.items():
            generator = agent_pair["generator"]
            evaluator = agent_pair["evaluator"]
            
            # Get last evaluator output
            last_generator_output = None
            for msg in reversed(generator.message):
                if msg["role"] == "assistant":
                    last_generator_output = msg["content"]
                    break
            
            saved_data[idx] = {
                "generator_messages": generator.message,
                "evaluator_messages": evaluator.message,
                "last_generator_output": last_generator_output
            }
        return saved_data


# ===== Test =====
# Test Generator class
def test_generator_initialization():
    generator = Generator("Test App", 3)
    assert generator.status == AgentStatus.BEGIN
    assert generator.request_counter == 0

def test_generate_prompt_formatting():
    generator = Generator("Test Description", 3)
    init_block = "User_constant_0 = 5"
    program = "API.call(param=User_constant_0)"
    
    prompt = generator.get_generate_prompt(init_block, program)
    assert "Test Description" in prompt
    assert init_block in prompt
    assert program in prompt

def test_response_parsing():
    generator = Generator("", 3)
    test_response = '''<User Variable Definition>
    user_variable_0 is a dict
    <Task Instructions>
    Create something'''
    
    parsed = generator.analyze_output(test_response)
    assert parsed["init_info"] == "user_variable_0 is a dict"
    assert parsed["description"] == "Create something"

# Test Evaluator class
def test_evaluator_decision_making():
    evaluator = Evaluator("", 3)
    assert evaluator.whether_continue("<OK>") is False
    assert evaluator.whether_continue("Needs improvement") is True

def test_evaluator_prompt_generation():
    evaluator = Evaluator("", 3)
    prompt = evaluator.get_evaluate_prompt(
        init_block="init",
        program="program steps",
        description="test description"
    )
    assert "init" in prompt
    assert "program steps" in prompt
    assert "test description" in prompt

# Test state transitions
def test_generator_state_flow():
    generator = Generator("", 3)
    generator.interact({"init_block": "", "program": ""})
    assert generator.status == AgentStatus.CONTINUE
    assert generator.request_counter == 1

def test_evaluator_state_transitions():
    evaluator = Evaluator("", 3)
    evaluator.interact({
        "init_block": "",
        "program": "",
        "description": ""
    })
    assert evaluator.status == AgentStatus.CONTINUE
    assert evaluator.request_counter == 1
# ===== Test =====


