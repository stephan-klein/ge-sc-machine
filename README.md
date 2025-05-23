# MANDO-GURU: Vulnerability Detection for Smart Contract Source Code By Heterogeneous Graph Embeddings - API Services
<p>
    <a href="https://www.python.org/" target="blank_"><img alt="python" src="https://img.shields.io/badge/python-3.8.9-green" /></a>
    <a href="https://fastapi.tiangolo.com/" target="blank_"><img alt="FastAPI" src="https://img.shields.io/badge/FastAPI-0.78.0-yellowgreen" /></a>
    <a href="https://reactjs.org/" target="blank_"><img alt="bigquery" src="https://img.shields.io/badge/ReactJs-18.2.0-red" /></a>
    <a href="https://opensource.org/licenses/MIT" target="blank_"><img alt="mit" src="https://img.shields.io/badge/License-MIT-blue.svg" /></a>
</p>
<p align="center">
<img src="./assets/MANDO_GURU.png" alt="MANDO GURU Logo" width="225" height="225" class="center">
</p>

# About the Fork
This project was forked from [Mando-Project](https://github.com/MANDO-Project/ge-sc-machine) to evaluate the model independently (See https://github.com/stephan-klein/mlsc)

It includes minor infrastructural changes to be able to run it in 2025.

List of changes:
- [Smartbugs](https://github.com/smartbugs/smartbugs) integration
  - Add command line scripts `inference.py` / `inference.sh` using logic from nodetype endpoint but providing some additional processing option. Use `inference.sh --help` for details.
  - Dockerfile: Remove entrypoint, it will be defined by smartbugs
- Frontend: In react app's package.json lock certain older dependencies of react-force-graph s.t `yarn start` does not end in an ECMAScript Error

# Overview
This project was created to bring out the APIs of vulnerability detection for smart contracts based on heterogeneous graph embeddings.
<p align="center">
<img src="./assets/Overview.png" alt="MANDO-GURU overview" class="center">
</p>

## Components
### Backend
- Backend plays a vital role with several core sub-components such as heterogeneous presentation for the generated graphs from input smart contract files, heterogeneous graph fusion, custom multi-metapaths extraction, heterogeneous graph neural network, and vulnerability detections in coarse-grained and fine-grained levels.
### Frontend
- Frontend component services are used to visualize the prediction results and the statistics of the analyzed smart contracts.

### RESTful APIs
- RESTful APIs are implemented as a bridge to communicate between the Backend and the Frontend.

## Citation
Nguyen, H. H., Nguyen, N. M., Doan, H. P., Ahmadi, Z., Doan, T. N., & Jiang, L. (2022, November). *MANDO-GURU: vulnerability detection for smart contract source code by heterogeneous graph embeddings.* In Proceedings of the 30th ACM Joint European Software Engineering Conference and Symposium on the Foundations of Software Engineering (pp. 1736-1740). [Preprint](https://hoanghnguyen.com/assets/pdf/nguyen2022fse.pdf)

```
@inproceedings{nguyen2022fse,
  author = {Nguyen, Hoang H. and Nguyen, Nhat-Minh and Doan, Hong-Phuc and Ahmadi, Zahra and Doan, Thanh-Nam and Jiang, Lingxiao},
  title = {MANDO-GURU: Vulnerability Detection for Smart Contract Source Code By Heterogeneous Graph Embeddings},
  year = {2022},
  month = {11},
  isbn = {9781450394130},
  publisher = {Association for Computing Machinery},
  address = {New York, NY, USA},
  url = {https://doi.org/10.1145/3540250.3558927},
  doi = {10.1145/3540250.3558927},
  booktitle = {Proceedings of the 30th ACM Joint European Software Engineering Conference and Symposium on the Foundations of Software Engineering},
  pages = {1736-1740},
  numpages = {5},
  keywords = {Ethereum blockchain, graph neural networks, heterogeneous graphs, smart contracts, vulnerability detection},
  location = {Singapore, Singapore},
  series = {ESEC/FSE 2022}
}
```

# Table of contents
- [MANDO-GURU services](#mando-guru-services)
- [Overview](#overview)
  - [Components](#components)
    - [Backend](#backend)
    - [Frontend](#frontend)
    - [RESTful APIs](#restful-apis)
- [Table of contents](#table-of-contents)
- [How to use the tool?](#how-to-use-the-tool)
  - [Coarse-Grained Detection](#coarse-grained-detection)
  - [Fine-Grained Detection](#fine-grained-detection)
  - [Statistics](#statistics)
  - [MandoGuru APIs](#mandoguru-apis)
  - [MandoGuru APIs' Token](#mandoguru-apis-token)
  - [Demo Video](#demo-video)
- [Deployment](#deployment)
  - [Prerequisites](#prerequisites)
  - [Deploy on Local Machine](#deploy-on-local-machine)
    - [Backend](#backend-1)
    - [Frontend](#frontend-1)

# How to use the tool?
## Coarse-Grained Detection
![GE-SC overview](./assets/onClickDetail.png)
- You can upload a solidity smart contract from local or select an available one in the drop-down box.
- We supported 7 kind of bugs. Due to limits of computation resources, the scanning process might take 1-2 minutes to get the results, depending on the complexity of the input source file. We recommend users use the Chrome browser for the best experience.
- When the process finished:
    - Red Button: Smart contract contains this type of bug.
    - Green Button: Smart contract does not contain this type of bug.

## Fine-Grained Detection
![GE-SC overview](./assets/mando-detection-screenshot.png)
- When you click a button which was result of Coaser-Grained phase, the tool would show the source code and graph of the smart contract
- Source Code
    - Buggy Code Line : The line of code has the background color of yellow.
- Graph 
    - Red Node: Bug Node.
    - White Node : Clean Node.
    - Border of Node : Node Type.
- If you click a node in the graph, the lines of code equivalent to that node will be bounded by a red border.

## Statistics
- Bar Chart
![GE-SC overview](./assets/BarChart.png)
  - Number of bug nodes and Number of clean nodes for each type of bug.
- Detection Time
![GE-SC overview](./assets/DetectionTime.png)
  - DetectionTime for each type of bug.
- Bug Density
![GE-SC overview](./assets/BugDensity.png)
  - We divided the line number which had bugs into 15 categories in order.
  - The portion with darker color shows that the areas of source code have more bug lines.


## MandoGuru APIs
- APIs documents: [mandoguru.com/docs](http://mandoguru.com/docs)
- We also published APIs documents for user can directly request to MandoGuru services.
- There are 2 main APIs:
  - Coarse-grained detection.
  - Fine-grained detection.
- The document page were built based on [Swagger](https://swagger.io/tools/swagger-ui/) which help you request directly.
- When making a request, **you have to authorize by [the public token](#mandoguru-apis-token) fisrt.**
- To call an API step by step, please refer to [demo video](http://mandoguru.com/demo-video).

## MandoGuru APIs' Token
```
MqQVfJ6Fq1umZnUI7ZuaycciCjxi3gM0
```

## Demo Video
Please visit this link to see the [demo video](http://mandoguru.com/demo-video).

# Deployment
- If you want to launch our tool yourselves, please meet the prerequisites prior to follow the steps bellow:

## Prerequisites
- [docker](https://docs.docker.com/engine/install/)
- [node](https://nodejs.org/en/download/) 16.15.1 or higher
- [npm](https://docs.npmjs.com/downloading-and-installing-node-js-and-npm) 8.11.0 or higher

## Deploy on Local Machine

### Backend
- We published docker image for launching backend service.

- Pull docker image from docker hub.
```
docker pull nguyenminh1807/sco:latest
```

- Run container and map port 5555:xxxx to any port you want to public (we used the same port over here).
```
docker run -it -d --rm  --name sco_app -p 5555:5555 nguyenminh1807/sco:latest
```

### Frontend
- You need to navigate to frontend directory first.
```
cd sco_frontend
```

- Install required package with legacy peer dependencies option.
```
npm install --legacy-peer-deps
```

- Launch app from local.
```
npm start
```
