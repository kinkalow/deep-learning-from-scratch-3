import weakref
import numpy as np
import contextlib


# =============================================================================
# Config
# =============================================================================
class Config:
    enable_backprop = True


@contextlib.contextmanager
def using_config(name, value):
    # 前処理(設定の変更)
    old_value = getattr(Config, name)
    setattr(Config, name, value)
    try:
        yield  # 呼び出し元に返す(メイン処理)
    finally:
        setattr(Config, name, old_value)  # 後処理(元に戻す)


def no_grad():
    return using_config('enable_backprop', False)


# =============================================================================
# Variable / Function
# =============================================================================
class Variable:
    # __array_priority__=0の場合、期待はずれの結果になる
    # Variable.__array_priority__ = 200; print(np.array([1, 2]) + Variable(np.array(3)))  # variable([4 5])
    # Variable.__array_priority__ = 0;   print(np.array([1, 2]) + Variable(np.array(3)))  # [variable(4) variable(5)]
    __array_priority__ = 200

    def __init__(self, data, name=None):
        if data is not None:
            if not isinstance(data, np.ndarray):
                raise TypeError('{} is not supported'.format(type(data)))

        self.data = data
        self.name = name
        self.grad = None
        self.creator = None
        self.generation = 0

    @property
    def shape(self):
        return self.data.shape

    @property
    def ndim(self):
        return self.data.ndim

    @property
    def size(self):
        return self.data.size

    @property
    def dtype(self):
        return self.data.dtype

    def __len__(self):
        return len(self.data)

    def __repr__(self):
        if self.data is None:
            return 'variable(None)'
        p = str(self.data).replace('\n', '\n' + ' ' * 9)
        return 'variable(' + p + ')'

    def set_creator(self, func):
        self.creator = func
        self.generation = func.generation + 1  # 出力側の世代

    def cleargrad(self):
        self.grad = None

    def backward(self, retain_grad=False):
        # 微分値(grad)の初期設定
        if self.grad is None:
            self.grad = np.ones_like(self.data)

        # 関数リストへ登録

        funcs = []
        seen_set = set()

        # 関数リスト内の世代が小さい順で並ぶように関数を並び替える
        def add_func(f):
            if f not in seen_set:
                funcs.append(f)
                seen_set.add(f)
                funcs.sort(key=lambda x: x.generation)

        add_func(self.creator)

        # 後方処理
        while funcs:
            # 入力側の微分値を取得
            f = funcs.pop()
            gys = [output().grad for output in f.outputs]  # output is weakref
            gxs = f.backward(*gys)  # ndarrayを渡し、微分値を得る
            if not isinstance(gxs, tuple):
                gxs = (gxs,)

            for x, gx in zip(f.inputs, gxs):
                # 入力側の微分値を更新
                if x.grad is None:
                    x.grad = gx
                else:
                    x.grad = x.grad + gx
                # 生成関数のリストを更新
                if x.creator is not None:
                    add_func(x.creator)

            # 微分値を破棄
            if not retain_grad:
                for y in f.outputs:
                    y().grad = None  # y is weakref


def as_variable(obj):
    if isinstance(obj, Variable):
        return obj
    return Variable(obj)


def as_array(x):
    if np.isscalar(x):
        return np.array(x)
    return x


class Function:
    def __call__(self, *inputs):
        inputs = [as_variable(x) for x in inputs]

        # 前方処理
        xs = [x.data for x in inputs]
        ys = self.forward(*xs)  # ndarrayを渡す
        if not isinstance(ys, tuple):
            ys = (ys,)
        outputs = [Variable(as_array(y)) for y in ys]  # ndarray => Variable

        # 入力変数、出力変数、世代を保存
        if Config.enable_backprop:
            # 世代を更新
            self.generation = max([x.generation for x in inputs])  # 関数の世代
            for output in outputs:
                output.set_creator(self)  # 生成関数を登録し、出力側の世代も更新
            # 変数を保存
            self.inputs = inputs
            self.outputs = [weakref.ref(output) for output in outputs]  # 出力変数と生成関数の関係が循環参照 => output:弱参照扱い

        # リストまたはVariable変数を返す
        return outputs if len(outputs) > 1 else outputs[0]

    def forward(self, xs):
        raise NotImplementedError()

    def backward(self, gys):
        raise NotImplementedError()


# =============================================================================
# 四則演算 / 演算子のオーバーロード
# =============================================================================
class Add(Function):
    def forward(self, x0, x1):
        y = x0 + x1
        return y

    def backward(self, gy):
        return gy, gy


def add(x0, x1):
    x1 = as_array(x1)
    return Add()(x0, x1)


class Mul(Function):
    def forward(self, x0, x1):
        y = x0 * x1
        return y

    def backward(self, gy):
        x0, x1 = self.inputs[0].data, self.inputs[1].data
        return gy * x1, gy * x0


def mul(x0, x1):
    x1 = as_array(x1)
    return Mul()(x0, x1)


class Neg(Function):
    def forward(self, x):
        return -x

    def backward(self, gy):
        return -gy


def neg(x):
    return Neg()(x)


class Sub(Function):
    def forward(self, x0, x1):
        y = x0 - x1
        return y

    def backward(self, gy):
        return gy, -gy


def sub(x0, x1):
    x1 = as_array(x1)
    return Sub()(x0, x1)


def rsub(x0, x1):  # x0:右辺項、x1:左辺項
    x1 = as_array(x1)
    return Sub()(x1, x0)


class Div(Function):
    def forward(self, x0, x1):
        y = x0 / x1
        return y

    def backward(self, gy):
        x0, x1 = self.inputs[0].data, self.inputs[1].data
        gx0 = gy / x1
        gx1 = gy * (-x0 / x1 ** 2)
        return gx0, gx1


def div(x0, x1):
    x1 = as_array(x1)
    return Div()(x0, x1)


def rdiv(x0, x1):
    x1 = as_array(x1)
    return Div()(x1, x0)


class Pow(Function):
    def __init__(self, c):
        self.c = c

    def forward(self, x):
        y = x ** self.c
        return y

    def backward(self, gy):
        x = self.inputs[0].data
        c = self.c

        gx = c * x ** (c - 1) * gy
        return gx


def pow(x, c):
    return Pow(c)(x)


def setup_variable():
    Variable.__add__ = add
    Variable.__radd__ = add
    Variable.__mul__ = mul
    Variable.__rmul__ = mul
    Variable.__neg__ = neg
    Variable.__sub__ = sub
    Variable.__rsub__ = rsub
    Variable.__truediv__ = div
    Variable.__rtruediv__ = rdiv
    Variable.__pow__ = pow
