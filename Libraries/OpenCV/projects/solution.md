# Solution notes

The contract prevents the common BGR/RGB and dtype/range mistakes before model
inference. Production code also bounds decoded dimensions, records codec and
input hashes, and tests annotation geometry through every transform.
