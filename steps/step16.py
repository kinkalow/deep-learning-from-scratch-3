import numpy as np


class Variable:
    def __init__(self, data):
        if data is not None:
            if not isinstance(data, np.ndarray):
                raise TypeError('{} is not supported'.format(type(data)))

        self.data = data
        self.grad = None
        self.creator = None
        self.generation = 0

    def set_creator(self, func):  # 出力側から呼ばれる
        self.creator = func
        self.generation = func.generation + 1

    def cleargrad(self):
        self.grad = None

    def backward(self):
        # 微分値の初期設定
        if self.grad is None:
            self.grad = np.ones_like(self.data)

        # 関数の世代を小さい順に並び替える

        funcs = []
        seen_set = set()

        def add_func(f):
            if f not in seen_set:
                funcs.append(f)
                seen_set.add(f)
                funcs.sort(key=lambda x: x.generation)

        add_func(self.creator)

        # 後方処理
        while funcs:
            f = funcs.pop()
            gys = [output.grad for output in f.outputs]  # 出力側の微分値をリスト化
            gxs = f.backward(*gys)                       # 入力側の微分値を得る
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


def as_array(x):
    if np.isscalar(x):
        return np.array(x)
    return x


class Function:
    def __call__(self, *inputs):  # inputsの要素: Variable変数

        # 前方処理
        xs = [x.data for x in inputs]
        ys = self.forward(*xs)
        if not isinstance(ys, tuple):
            ys = (ys,)

        # 世代の更新
        self.generation = max([x.generation for x in inputs])  # 入力側の世代
        outputs = [Variable(as_array(y)) for y in ys]
        for output in outputs:
            output.set_creator(self)  # 生成関数の設定に加えて、出力側の世代も更新する

        # 保存
        self.inputs = inputs
        self.outputs = outputs

        return outputs if len(outputs) > 1 else outputs[0]  # リストまたはVariable変数を返す

    def forward(self, xs):
        raise NotImplementedError()

    def backward(self, gys):
        raise NotImplementedError()


class Square(Function):
    def forward(self, x):
        y = x ** 2
        return y

    def backward(self, gy):
        x = self.inputs[0].data
        gx = 2 * x * gy
        return gx


def square(x):
    return Square()(x)


class Add(Function):
    def forward(self, x0, x1):
        y = x0 + x1
        return y

    def backward(self, gy):
        return gy, gy


def add(x0, x1):
    return Add()(x0, x1)


x = Variable(np.array(2.0))
a = square(x)
y = add(square(a), square(a))
y.backward()

print(y.data)
print(x.grad)
