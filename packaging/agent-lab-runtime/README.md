# Remy Agent Lab container runtime

This image contains only CPython and the Alpine base. Agent Lab adds the actual
security boundary at `docker run`: no network, no image pull, read-only rootfs,
all capabilities dropped, no-new-privileges, bounded PID/RAM/CPU resources, a
non-root UID, bounded tmpfs, and one run-owned workspace mount.

The desktop installer bundles this small build context. In Agent Lab, the user
can explicitly select **Prepare secure runtime**. This may download the pinned
CPython/Alpine base from Docker Hub when it is not already local; it never needs
an account or registry credentials. The same operation can be run from source:

```powershell
docker build --pull=false -t remy-agent-lab-runtime:py312 packaging/agent-lab-runtime
```

Remy never starts preparation in the background. Agent Lab execution itself
continues to use `--pull=never` and never registers with a container registry.

The runtime preflight resolves the tag to an immutable local `sha256:` image ID,
rejects remote Docker contexts, and uses that ID for every execution.
