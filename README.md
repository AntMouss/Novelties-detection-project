## Installation

### Prerequisite

if you are on Ubuntu 20.04 , you can run the `linux-script-installation.bash` as refered [here](#with-bash) . 
else you can use docker , refered to this [section](#with-docker) but first you need to install docker-engine on
your machine . The installation steps of docker engine for your operating systems might be slightly different, please refer to the [docker documentation](https://docs.docker.com/engine/install/) for details.

### With Bash

```bash
#first download repo from github and unzip.
$ wget https://github.com/AntMouss/Novelties-detection-project/archive/master.zip -O novelties-detection-master.zip
$ unzip novelties-detection-master.zip

#changes current directory to novelties-detection-master.
$ cd novelties-detection-master

#give authorization for excecuting linux-script-installation.bash
$ chmod x+u linux-script-installation.bash

# run linux-script-installation.bash with ouput_path arg.
$ linux-script-installation.bash <output_path>

```
if you don't specify `output_path` the collect service will not persistent. 
### With Docker

```bash
# build image from github repo.
$ docker build --tag novelties-detection-image https://github.com/AntMouss/Novelties-detection-project.git

# run container from the image that we build previously.
$ docker run -d -p 5000:5000 \
  --name <container_name> \
  --mount source=<volume_name>,target=/collect_data \
  -e OUTPUT_PATH=/collect_data \
  novelties-detection-image:latest

# to check the logs from the container ,
# use this command with the same container_name of the command above.
$ docker logs <container_name>

# you can access the volume data with this command if you are on Ubuntu with sudo privilege.
$ sudo ls /var/lib/docker/volumes/<volume_name>/_data

```