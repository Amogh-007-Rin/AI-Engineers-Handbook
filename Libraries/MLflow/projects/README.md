# MLflow run-evidence project

Validate a dependency-free run manifest and model-promotion gate before wiring
it to a tracking server. In the declared environment, the native test uses a
real MLflow 3.x SQLite tracking backend and explicit local artifact store to
create an experiment/run, persist provenance tags and a metric, log a signature
artifact, terminate the run, read it back, and apply the promotion gate. This
avoids MLflow 3's maintenance-only legacy filesystem tracking backend. Run
`python -W error -m unittest -v`; extend it with registry lineage, signature
enforcement, access controls, and rollback.
