
Reference: [Real Time Inference on Raspberry Pi 4](https://pytorch.org/tutorials/intermediate/realtime_rpi.html)

## Setup Raspberry Pi

 (aarch64) so you’ll need to install a 64 bit version of the OS on your Raspberry Pi

You can download the latest arm64 Raspberry Pi OS from https://downloads.raspberrypi.org/raspios_arm64/images/ and install it via rpi-imager.

See the following official Raspberry Pi documentation for more information on how to setup your Raspberry Pi.
* [Connect a camera module](https://projects.raspberrypi.org/en/projects/getting-started-with-picamera)
* [Install Arm 64bit OS](https://www.raspberrypi.com/documentation/computers/getting-started.html#raspberry-pi-imager): PyTorch only provides pip packages for Arm 64bit
* [Remote access](https://www.raspberrypi.com/documentation/computers/remote-access.html)
    - `ssh your-id@raspberrypi -L 8888:localhost:8888`


/boot/config.txt
```
# This enables the extended features such as the camera.
start_x=1

# This needs to be at least 128M for the camera processing, if it's bigger you can just leave it as is.
gpu_mem=128

# You need to comment/remove the existing camera_auto_detect line since this causes issues with OpenCV/V4L2 capture.
#camera_auto_detect=1
```

And then reboot. After you reboot the video4linux2 device /dev/video0 should exist.


### Setup python dependencies

```bash
$ python -m venv ./env/torch
$ source ./env/torch/bin/activate
$ pip install torch torchvision torchaudio
$ pip install opencv-python
```
