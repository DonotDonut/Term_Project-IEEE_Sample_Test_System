# Newton-Raphson Power Flow Analysis 

This repository contains a **Python** implementation of the **Newton-Raphson (NR) power flow analysis**. The program is designed to take input from the **[IEEE Power Flow Test Cases](https://lamarr.ece.uw.edu/research/pstca)** and compute key power flow results such as bus voltage magnitudes and angles, generator outputs, and line flows.

Below is the flowchart illustrating the Newton-Raphson power flow calculation process:

![Term Project Flowchart](image/image.png)

### Future Works 
- Improve the Jacobian matrix
- improve the Qlimits enforcement
- Include shunt susceptance to certain bus
- include transformer phase shift angle

### Prerequisites
Ensure you have Python 3.x installed, along with the following libraries:
```bash
pip install numpy

### License
Anyone can use the code if you would like to help improve this code fork it 
