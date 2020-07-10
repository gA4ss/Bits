#!/usr/bin/python
# coding:utf-8
import numpy as np

class Bits:
    def __init__(self, fields=[], names=[], n=64):
        # FIXME 这里没有验证参数的长度，类型信息， fields的长度应该与names的长度一致

        self.fields = np.array([])
        for field in fields:
            if type(field) is not int:
                raise TypeError("[error] Bit field must be integer type")
            self.fields = np.append(self.fields, field)
        self.names = names

        if np.sum(self.fields) > n:
            raise ValueError("[error] Total of bits > %d" % (n))

        self.bits = n

        c = 0
        self.protocols = []
        self.protocols_names = {}
        for i, b in enumerate(fields):
            protocol = {}
            protocol['shift'] = n - b - c
            protocol['mask'] = (2**b << protocol['shift'])-1
            protocol['size'] = 2**b-1
            self.protocols.append(protocol)
            c += b
            # 添加名称属性
            if len(names) != 0:
                self.protocols_names[names[i]] = protocol
        self.space = n - c

    def __getitem__(self, key):
        if type(key) is str:
            return self.protocols_names[key]
        # FIXME 类型验证不严格
        return self.protocols[key]

    def _decode(self, code, mask, rshift):
        b = (code & np.uint64(mask)) >> np.uint64(rshift)
        if self.bits == 8:
            return np.uint8(b)
        elif self.bits == 16:
            return np.uint16(b)
        elif self.bits == 32:
            return np.uint32(b)
        elif self.bits == 64:
            return np.uint64(b)
        return np.uint(b)

    def _encode(self, code, c, bits, lshift):
        b = code | np.uint64((c & bits) << lshift)
        if self.bits == 8:
            return np.uint8(b)
        elif self.bits == 16:
            return np.uint16(b)
        elif self.bits == 32:
            return np.uint32(b)
        elif self.bits == 64:
            return np.uint64(b)
        return np.uint(b)

    def decode(self, field, code):
        if code is None:
            code = 0
        code = np.uint64(code)
        # FIXME 这里对参数的类型验证不严格
        protocol = self.__getitem__(field)
        c = self._decode(code, protocol['mask'], protocol['shift'])
        return c

    def decodes(self, fields, code):
        # FIXME 这里对参数的类型验证不严格
        codes = {}
        for field in fields:
            c = self.decode(field, code)
            if type(field) is int:
                fname = self.names.index(field)
            else:
                fname = field
            codes[fname] = c
        return codes

    def encode(self, field, value, code=None):
        if code is None:
            code = 0
        code = np.uint64(code)
        # FIXME 这里对参数的类型验证不严格
        protocol = self.__getitem__(field)
        code = self._encode(code, value, protocol['size'], protocol['shift'])
        return code

    def encodes(self, fields=[], values=[], code=None):
        # FIXME 这里对参数的类型验证不严格
        if len(fields) != len(values):
            raise ValueError("[error] Fields length must equal Values length")
        for field, value in zip(fields, values):
            code = self.encode(field, value, code)
        return code


if __name__ == '__main__':

    _type_const = 0
    _type_input = 1
    _type_weight = 2
    _type_neure = 3
    _type_layer = 4

    _type_str = ['const', 'input', 'weight', 'neure', 'layer']

    _layer_type_conv = 0
    _layer_type_pooling = 1
    _layer_type_dropout = 2
    _layer_type_flatten = 3
    _layer_type_dense = 4

    _layer_type = ['conv', 'pooling', 'dropout', 'flatten', 'dense']
    _activations = ['linear', 'relu', 'sigmoid', 'tanh', 'leaky relu', 'softmax', 'custom']

    _acf_linear = 0
    _acf_relu = 1
    _acf_sigmod = 2
    _acf_tanh = 3
    _acf_leaky_relu = 4
    _acf_softmax = 5
    _acf_custom = 6

    _pooling_max = 10
    _pooling_average = 11
    _poolings = ['max', 'average']

    _cnn_head_code = Bits(fields=[3, 5, 56], names=['type', 'reserve', 'others'], n=64)
    _cnn_const_code = Bits(fields=[3, 5, 56], names=['type', 'reserve', 'value'], n=64)
    _cnn_input_code = Bits(fields=[3, 5, 28, 26, 2], names=['type', 'reserve', 'row', 'col', 'cn'], n=64)
    _cnn_weight_code = Bits(fields=[3, 5, 18, 18, 10, 10], names=['type', 'reserve', 'neure', 'wmat', 'row', 'col'], n=64)
    _cnn_neure_code = Bits(fields=[3, 5, 18, 4, 16, 16, 2], names=['type', 'param', 'neure', 'func', 'row', 'col', 'end'], n=64)
    _cnn_layer_code = Bits(fields=[3, 5, 4, 1, 1, 4, 4, 4, 4, 4, 8, 11, 11], \
        names=['type', 'reserve', 'ltype', 'output', 'end', 'krow', 'kcol', \
        'srow', 'scol', 'padding', 'pwhat', 'parents', 'index'], n=64)

    def _encode_c(value=0, code=None):
        return _cnn_const_code.encodes(_cnn_const_code.names, [_type_const, 0, value], code)

    def _encode_a(row=0, col=0, cn=0, code=None):
        return _cnn_input_code.encodes(_cnn_input_code.names, [_type_input, 0, row, col, cn], code)

    def _encode_w(neure=0, wmat=0, row=0, col=0, code=None):
        return _cnn_weight_code.encodes(_cnn_weight_code.names, [_type_weight, 0, neure, wmat, row, col], code)

    def _encode_n(neure=0, func=0, param=0, row=0, col=0, end=0, code=None):
        if type(func) is str:
            if func in _activations:
                func = _activations.index(func)
            elif func in _poolings:
                func = _poolings.index(func) + _pooling_max
            else:
                raise ValueError("[error] Encode: function name is not valid")
        return _cnn_neure_code.encodes(_cnn_neure_code.names, [_type_neure, param, neure, func, row, col, end], code)

    def _encode_l(ltype=_layer_type_conv, output=0, end=0, \
                krow=3, kcol=3, srow=1, scol=1, padding='VALID', pwhat=0, \
                parents=-1, index=0, code=None):
        """
        如果是padding是SAME 则设置 >1
        """
        padding = 0 if padding=='VALID' else padding
        parents = 2**11 if parents == -1 else parents
        p = [_type_layer, 0, ltype, output, end, krow, kcol, srow, scol, padding, pwhat, parents, index]
        return _cnn_layer_code.encodes(_cnn_layer_code.names, p, code)


    ############################## 解码工具定义 ##############################
    def cnncode_decode_type(code):
        dc = _cnn_head_code.decode('type', code)
        if dc < _type_const or dc > _type_layer:
            raise Exception("[error] Decode: invalid type")
        return _type_str[dc]

    def cnncode_decode_const_code(code):
        fields=['type', 'reserve', 'value']
        dc = _cnn_const_code.decodes(fields, code)
        if dc['type'] != _type_const:
            raise Exception("[error] Decode: not const type")
        return dc['value']

    def cnncode_decode_input_code(code):
        fields=['type', 'reserve', 'row', 'col', 'cn']
        dc = _cnn_input_code.decodes(fields, code)
        if dc['type'] != _type_input:
            raise Exception("[error] Decode: not input type")
        output = {}
        output['row'] = dc['row']
        output['col'] = dc['col']
        output['cn'] = dc['cn']
        return output

    def cnncode_decode_weight_code(code):
        fields=['type', 'reserve', 'neure', 'wmat', 'row', 'col']
        dc = _cnn_weight_code.decodes(fields, code)
        if dc['type'] != _type_weight:
            raise Exception("[error] Decode: not weight type")
        output = {}
        output['neure'] = dc['neure']
        output['wmat'] = dc['wmat']
        output['row'] = dc['row']
        output['col'] = dc['col']
        return output

    def cnncode_decode_neure_code(code):
        fields=['type', 'param', 'neure', 'func', 'row', 'col', 'end']
        dc = _cnn_neure_code.decodes(fields, code)
        if dc['type'] != _type_neure:
            raise Exception("[error] Decode: not neure type")
        output = {}
        output['neure'] = dc['neure']
        output['func'] = dc['func']
        output['param'] = dc['param']
        output['row'] = dc['row']
        output['col'] = dc['col']
        output['end'] = dc['end']
        return output

    def cnncode_decode_layer_code(code):
        fields=['type', 'reserve', 'ltype', 'output', 'end', 'krow', 'kcol', \
            'srow', 'scol', 'padding', 'pwhat', 'parents', 'index']
        dc = _cnn_layer_code.decodes(fields, code)
        if dc['type'] != _type_layer:
            raise Exception("[error] Decode: not layer type")
        output = {}
        output['ltype'] = dc['ltype']
        output['output'] = dc['output']
        output['end'] = dc['end']
        output['krow'] = dc['krow']
        output['kcol'] = dc['kcol']
        output['srow'] = dc['srow']
        output['scol'] = dc['scol']
        output['padding'] = dc['padding']
        output['pwhat'] = dc['pwhat']
        output['parents'] = dc['parents']
        output['index'] = dc['index']
        return output
else:
    pass