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

In your description, first define user variable/constant name (such as User_variable_0) in the init block with necessary descriptions of their types and structures (e.g., dict with what key), but do not directly show their values since they may change with respect to different test cases.
Provide necessary instructions when these variable/constant names are referred to. ONLY refer to names, not values. 
All user-provided content should be used (e.g., when the variable value matches User_constant_0) if they are referred to the program.
If there is RESULT variable, specify it as its vaule will be checked using test cases.

Response Format:
<User Variable Definition>
User_variable_0 is a xxx, ...
<Task Instructions>
Implement ...
'''

GENERATOR_IMPROVE_PROMPT = ''' An evaluator agent has checked the descriptions and offered some advices. Please improve your description:
{evaluator_output}
'''

EVALUATOR_PROMPT = '''
You are a checker agent tasked with evaluating the quality of a natural language description that was generated from a sequence of API calls. Your job is to determine whether the description meets two key criteria:

1.	Fidelity to the Program Logic
	•	Does the natural language description accurately reflect all steps of the given API call sequence?
	•	Would a developer be able to reconstruct the original program logic (or a functionally equivalent one) based solely on the description?
	•	Are any steps missing, incorrectly described, or added?
	• Is there any ambiguity? For example, sometimes "update" can refer to either "local update" or "remote update through APIs." 
2.	Human-likeness and Fluency
	•	Does the description sound natural and fluent, as if it were written by a human programmer?
	•	Does it frame the task in a realistic scenario, avoiding robotic or overly literal translations

Your output should be one of the following:
	•	If the description satisfies both criteria: <OK>
	•	If there are any problems, output a short diagnosis and suggestions for improvement. Be concise but specific.

Below is the program:
```
# ===== Init Block Begin =====
{init_block}
# ===== Init Block End =====

{program}
```

Below is the description:
{description}
'''

EVALUATOR_FURTHER_PROMPT = '''
Here are the updated descriptions according to your suggestions:
{description}

If the description satisfies all the criteria mentioned above, generate: <OK>
Otherwise, output a short diagnosis and suggestions for improvement. Be concise but specific.
'''