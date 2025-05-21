## conv2d
torch.nn.functional.conv2d(input, weight, bias=None, stride=1, padding=0, dilation=1, groups=1) -> Tensor

Applies a 2D convolution over an input image composed of several input planes.


Args:
    input: input tensor of shape :math:`(\text{minibatch} , \text{in\_channels} , iH , iW)`
    weight: filters of shape :math:`(\text{out\_channels} , \frac{\text{in\_channels}}{\text{groups}} , kH , kW)`
    bias: optional bias tensor of shape :math:`(\text{out\_channels})`. Default: ``None``
    stride: the stride of the convolving kernel. Can be a single number or a
      tuple `(sH, sW)`. Default: 1
    padding: implicit paddings on both sides of the input. Can be a string {'valid', 'same'},
      single number or a tuple `(padH, padW)`. Default: 0
      ``padding='valid'`` is the same as no padding. ``padding='same'`` pads
      the input so the output has the same shape as the input. However, this mode
      doesn't support any stride values other than 1.

      .. warning::
          For ``padding='same'``, if the ``weight`` is even-length and
          ``dilation`` is odd in any dimension, a full :func:`pad` operation
          may be needed internally. Lowering performance.

    dilation: the spacing between kernel elements. Can be a single number or
      a tuple `(dH, dW)`. Default: 1

Shape:
    - Input: :math:`(N, C_{in}, H_{in}, W_{in})`
    - Output: :math:`(N, C_{out}, H_{out}, W_{out})`

        .. math::
            H_{out} = \left\lfloor\frac{H_{in}  + 2 \times \text{padding}[0] - \text{dilation}[0]
                    \times (\text{kernel\_size}[0] - 1) - 1}{\text{stride}[0]} + 1\right\rfloor

        .. math::
            W_{out} = \left\lfloor\frac{W_{in}  + 2 \times \text{padding}[1] - \text{dilation}[1]
                    \times (\text{kernel\_size}[1] - 1) - 1}{\text{stride}[1]} + 1\right\rfloor

Examples::
"""
    >>> # With square kernels and equal stride
    >>> filters = torch.randn(8, 4, 3, 3)
    >>> inputs = torch.randn(1, 4, 5, 5)
    >>> F.conv2d(inputs, filters, padding=1)
"""

---

## permute
torch.permute(input, dims) → Tensor

Returns a view of the original tensor input with its dimensions permuted.

Parameters
    input (Tensor) – the input tensor.
    dims (tuple of int) – The desired ordering of dimensions

Example::
"""
    >>> x = torch.randn(2, 3, 5)
    >>> x.size()
    torch.Size([2, 3, 5])
    >>> torch.permute(x, (2, 0, 1)).size()
    torch.Size([5, 2, 3])
"""

---

## split
torch.split(tensor, split_size_or_sections, dim=0)

Splits the tensor into chunks. Each chunk is a view of the original tensor.

If :attr:`split_size_or_sections` is an integer type, then :attr:`tensor` will
be split into equally sized chunks (if possible). Last chunk will be smaller if
the tensor size along the given dimension :attr:`dim` is not divisible by
:attr:`split_size`.

If :attr:`split_size_or_sections` is a list, then :attr:`tensor` will be split
into ``len(split_size_or_sections)`` chunks with sizes in :attr:`dim` according
to :attr:`split_size_or_sections`.

Args:
    tensor (Tensor): tensor to split.
    split_size_or_sections (int) or (list(int)): size of a single chunk or
        list of sizes for each chunk
    dim (int): dimension along which to split the tensor.

Example::
"""
    >>> a = torch.arange(10).reshape(5, 2)
    >>> a
    tensor([[0, 1],
            [2, 3],
            [4, 5],
            [6, 7],
            [8, 9]])
    >>> torch.split(a, 2)
    (tensor([[0, 1],
                [2, 3]]),
        tensor([[4, 5],
                [6, 7]]),
        tensor([[8, 9]]))
    >>> torch.split(a, [1, 4])
    (tensor([[0, 1]]),
        tensor([[2, 3],
                [4, 5],
                [6, 7],
                [8, 9]]))
"""

---

## cat
torch.cat(tensors, dim=0, *, out=None) → Tensor

Concatenates the given sequence of tensors in tensors in the given dimension. All tensors must either have the same shape (except in the concatenating dimension) or be a 1-D empty tensor with size (0,).
torch.cat() can be seen as an inverse operation for torch.split() and torch.chunk().
torch.cat() can be best understood via examples.

Parameters
    tensors (sequence of Tensors) – Non-empty tensors provided must have the same shape, except in the cat dimension.
    dim (int, optional) – the dimension over which the tensors are concatenated

Keyword Arguments
    out (Tensor, optional) – the output tensor.

Example::
"""
    >>> x = torch.randn(2, 3)
    >>> x
    tensor([[ 0.6580, -1.0969, -0.4614],
            [-0.1034, -0.5790,  0.1497]])
    >>> torch.cat((x, x, x), 0)
    tensor([[ 0.6580, -1.0969, -0.4614],
            [-0.1034, -0.5790,  0.1497],
            [ 0.6580, -1.0969, -0.4614],
            [-0.1034, -0.5790,  0.1497],
            [ 0.6580, -1.0969, -0.4614],
            [-0.1034, -0.5790,  0.1497]])
    >>> torch.cat((x, x, x), 1)
    tensor([[ 0.6580, -1.0969, -0.4614,  0.6580, -1.0969, -0.4614,  0.6580,
            -1.0969, -0.4614],
            [-0.1034, -0.5790,  0.1497, -0.1034, -0.5790,  0.1497, -0.1034,
            -0.5790,  0.1497]])
"""

---

## linear
torch.nn.functional.linear(input, weight, bias=None) -> Tensor

Applies a linear transformation to the incoming data: :math:`y = xA^T + b`.

Shape:
    - Input: :math:`(*, in\_features)` where `*` means any number of
      additional dimensions, including none
    - Weight: :math:`(out\_features, in\_features)` or :math:`(in\_features)`
    - Bias: :math:`(out\_features)` or :math:`()`
    - Output: :math:`(*, out\_features)` or :math:`(*)`, based on the shape of the weight

## transpose
torch.transpose(input, dim0, dim1) -> Tensor
    
Returns a tensor that is a transposed version of :attr:`input`.
The given dimensions :attr:`dim0` and :attr:`dim1` are swapped.

Args:
    input (Tensor): the input tensor.
    dim0 (int): the first dimension to be transposed
    dim1 (int): the second dimension to be transposed

Example::
"""
    >>> x = torch.randn(2, 3)
    >>> x
    tensor([[ 1.0028, -0.9893,  0.5809],
            [-0.1669,  0.7299,  0.4942]])
    >>> torch.transpose(x, 0, 1)
    tensor([[ 1.0028, -0.1669],
            [-0.9893,  0.7299],
            [ 0.5809,  0.4942]])
"""