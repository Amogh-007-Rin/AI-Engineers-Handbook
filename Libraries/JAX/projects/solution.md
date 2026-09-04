# JAX solution guidance

The reference keeps loss and step pure, represents parameters as a pytree, uses `value_and_grad`, compiles one fixed-shape step, and validates gradients independently. The extension must return all state explicitly, split random keys, separate compile timing, and use framework-native serialization with schema/version metadata.
