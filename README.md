## Repository for sume cool visualizations I end up making from my research/personal projects
Disclaimer: This repo does use functions and code written from other methods. I try to cite or point them out as much as possible, but some might be missed. A lot of this code is just me playing around with different visualization methods.

### Scripts
`acoustic_projection.py`:
- This script is super messy, and I wrote the quick `generate_example_data.py` without thinking too much so that I could just generate data to visualize. This script is from another repo that I'm working on so the structure has been morphed from that repo into being able to run by itself, so just not the messiness of my code. If you run it, it works... thats all I offer.
- This script is meant to overlay the view frustum of an acoustic sensor over rays originating from a camera that uses the standard pinhole model. The idea is that the camera and the acoustic sensor overlap, and to just visualize it.
- The visualization below displays a simple one. The poses of the camera/acoustic sensor follow that of a spiral. The colorful disks represent the spherical frustum at a distance `d` from the acoustic sensor making the measurement. The size of the frustum is dictated by the view angle of the sensor, which is best determined through experimental work.
<p align='center'>
![Hehe](https://github.com/onurbagoren/visualizations/blob/main/media/spiral_frustum.gif)
</p>