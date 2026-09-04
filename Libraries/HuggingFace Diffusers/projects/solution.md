# Solution notes

Testing the scheduler alone makes timestep and shape behavior cheap and clear.
A full generation system pins every component, uses isolated generators, records
parameters/provenance, and evaluates both output quality and harm.
