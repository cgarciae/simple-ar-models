from flax.experimental import nnx
import jax.numpy as jnp
import jax
import optax


class AR(nnx.Module):
    def __init__(
        self, n_tokens: int, n_features: int, seq_len: int, *, ctx: nnx.Context
    ):
        self.n_tokens = n_tokens
        self.n_features = n_features
        self.seq_len = seq_len
        self.embed = nnx.Embed(n_tokens, n_features, ctx=ctx)
        self.kernel: jax.Array = nnx.Param(  # type: ignore
            nnx.initializers.lecun_normal()(
                ctx.make_rng("params"),
                (seq_len, seq_len, n_features, n_features),
                jnp.float32,
            )
        )

    def __call__(self, x: jax.Array, debug: bool = False) -> jax.Array:
        x = self.embed(x)
        mask = jnp.tril(jnp.ones((self.seq_len, self.seq_len)))
        if debug:
            jax.debug.print("mask = \n{mask}", mask=mask)
        x = jnp.einsum("...sd, tsdo, ts -> ...to", x, self.kernel, mask)
        x = self.embed.attend(x)
        return x


a = jnp.array([False, False, True, True])
b = jnp.array([False, True, False, True])
c = a & b
d = a | b
xor = a ^ b

X = jnp.stack([a, b, c, d, xor], axis=1, dtype=jnp.int32)
print(X)
inputs, labels = X[:, :-1], X[:, 1:]

model = AR(n_tokens=2, n_features=32, seq_len=4, ctx=nnx.context(0))
params, moduledef = model.partition(nnx.Param)
state = nnx.TrainState(moduledef, params=params, tx=optax.adam(1e-4))
model(inputs, debug=True)


@jax.jit
def train_step(
    state: nnx.TrainState[AR], inputs: jax.Array, labels: jax.Array
) -> nnx.TrainState[AR]:
    def loss_fn(params):
        logits, _ = state.apply(params)(inputs)
        loss = jnp.mean(optax.softmax_cross_entropy_with_integer_labels(logits, labels))
        return loss

    grads = jax.grad(loss_fn)(state.params)
    state = state.apply_gradients(grads)
    return state


@jax.jit
def test_step(
    state: nnx.TrainState[AR], inputs: jax.Array, labels: jax.Array
) -> tuple[jax.Array, jax.Array]:
    logits, _ = state.apply("params")(inputs)
    loss = jnp.mean(optax.softmax_cross_entropy_with_integer_labels(logits, labels))
    preds = sample(state, X.at[:, 2:].set(0))
    accuracy = jnp.mean(preds[:, -1] == labels[:, -1])
    return loss, accuracy


@jax.jit
def sample(state: nnx.TrainState[AR], output: jax.Array) -> jax.Array:
    for i in range(1, 4):
        logits, _ = state.apply("params")(output[:, :-1])
        tokens = jnp.argmax(logits[:, i], axis=-1)
        output = output.at[:, i + 1].set(tokens)

    return output


train_steps = 1_500
test_steps = 100

for i in range(train_steps + 1):
    state = train_step(state, inputs, labels)
    if i % 100 == 0:
        loss, accuracy = test_step(state, inputs, labels)
        print(f"[{i}] loss: {loss:.4f}, accuracy: {accuracy}")

model.update_state(state.params)


output = sample(state, X.at[:, 2:].set(0))
# animation
from rich.table import Table
import rich

print("\n\n")
columns = ["a", "b", "a&b", "a|b", "a^b"]

for i in range(2, 6):
    table = Table(title=f"Step {i-2}")
    for col in columns[:i]:
        table.add_column(col)

    for row in output:
        table.add_row(*[str(x) for x in row[:i]])

    rich.print(table)

print("\n\n")
