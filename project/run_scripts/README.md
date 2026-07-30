# Repository-managed Run Scripts

이 디렉터리가 BF-ODE-Edit의 tracked 실행 script와 wrapper의 canonical
location이다. root `run-scripts/`는 사용하지 않는다.

현재 실행 가능한 Stage 0 script는 아직 없다. server-head가 baseline source,
model access, dataset mount, Slurm GPU cap을 확인하고 red-team pre-flight를
통과한 뒤에만 `stage0_*` script를 추가할 수 있다. Script는 dataset,
checkpoint, raw generation, full log를 repository에 쓰지 않고 `local/` 경로를
받아야 한다.
