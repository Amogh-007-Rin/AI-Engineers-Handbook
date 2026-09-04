# Solution notes

The small model isolates the engineering contracts: gradients are checked
before mutation, the serving function accepts dynamic batches, and the test
invokes the exported signature by named input and output. A production solution
also records versions and compares held-out performance with a constant model.
