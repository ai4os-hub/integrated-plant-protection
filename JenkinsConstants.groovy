import groovy.transform.Field
// This file defines three variables to be used in the AI4OS-Hub Upstream Jenkins pipeline
// base_cpu_tag : base docker image for Dockerfile, CPU version
// base_gpu_tag : base docker image for Dockerfile, GPU version
// dockerfile : what Dockerfile to use for building, can include path, e.g. docker/Dockerfile

//@Field
//def base_cpu_tag = '2.1.2'

//@Field
//def base_gpu_tag = '2.1.2-cuda12.1-cudnn8-runtime'

@Field
def dockerfile = 'Dockerfile'

return this;