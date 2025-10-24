# TiRex Inference Server

This docker container runs the TiRex model and provides the following APIs to interact with the model:
- **HTTP API**
- **MQTT**

## Using the docker container

### HTTP API

Run the CPU container:
```
docker run -p 8000:8000 -it ghcr.io/nx-ai/tirex-cpu
```

Run the GPU container:
```
docker run --gpus all -p 8000:8000 -it ghcr.io/nx-ai/tirex-gpu
```

The GPU container has torch.compile enabled by default, so the model is compiled at startup. That might take over 2 minutes but leads to much faster request times afterwards.
To disable compilation, set the flag `-e MODEL_COMPILE=0` when running the container. You can also enable torch compile on the CPU, but that leads to slightly worse forecasts,
as the current version of compile on the CPU implements some operations differently.

When the container is running and has the model loaded, it exposes the HTTP API at [http://localhost:8000/](http://localhost:8000/). Swagger documentation of the API is provided at [http://localhost:8000/docs](http://localhost:8000/docs).

Bash:
```bash
curl -X 'POST' 'http://localhost:8000/forecast/mean' -H 'Content-Type: application/json' -d '{"context": [[0, 1, 2, 3]], "prediction_length": 4}'
````

Python:
```python
import requests
body = {"context": [[0, 1, 2, 3]], "prediction_length": 4}
response = requests.post('http://localhost:8000/forecast/mean', json=body)
print(response.text)
````

JavaScript/NodeJS:
```js
const body = JSON.stringify({ context: [[0, 1, 2, 3]], prediction_length: 4 });
const headers = { 'Content-Type': 'application/json' };
const data = await fetch('http://localhost:8000/forecast/mean', { method: 'POST', headers, body });
console.log(await data.json());
```

Every request is batched, so provide a list of timeseries as context, even when you only forecast a single timeseries. Bigger batch sizes are more efficient for the hardware, but too big batch sizes can lead to out of memory errors. There isn't any internal batching done, so the consumer of the API is responsible to call with an appropriate batch size for the hardware.

The HTTP API also provides a second endpoint `/forecast/quantiles`, where the 10, 20, 30, 50 (mean), 60, 70, 80 and 90% quantiles are returned, using the same arguments as the `/forecast/mean` endpoint.

### MQTT API
To use the MQTT API, you need an appropriate MQTT broker running. For some quick testing, a public test MQTT broker like [broker.emqx.io](https://broker.emqx.io) works (Do not send sensitive data to a public broker!). For testing we also use the [MQTTX CLI](https://mqttx.app/cli)

Start the container with MQTT:
```
docker run -p 8000:8000 -it -e MQTT_ENABLED=1 -e MQTT_BROKER_HOST=broker.emqx.io -e MQTT_BROKER_PORT=1883 ghcr.io/nx-ai/tirex-cpu
```

Subscribe to result topic:
```
mqttx sub -t 'tirex/forecast/result' -h 'broker.emqx.io' -p 1883
```

Send forecast request:
```
mqttx pub -t 'tirex/forecast/request' -h 'broker.emqx.io' -p 1883 -m '{"id": "1234", "context": [[0, 1, 2, 3]], "prediction_length": 4}'
```

If an error happens during processing the request, that error is published to the topic `tirex/forecast/error`.

## Configuration Options

You can set these env variables when running the container using the -e env flag, like `docker run -e MODEL_COMPILE=0 ghcr.io/nx-ai/tirex-cpu`

| Environment Variable           | Default Value            | Description                                                     |
| :----------------------------- | :----------------------- | :-------------------------------------------------------------- |
| **MODEL_PATH**                 | `NX-AI/TiRex`            | The Huggingface model id.                                       |
| **MODEL_COMPILE**              | `0` for CPU, `1` for GPU | torch.compile on the model. (1=True, 0=False)                   |
| **MQTT_ENABLED**               | `0`                      | Enable MQTT client functionality (1=True, 0=False)              |
| **MQTT_BROKER_HOST**           | `None`                   | Hostname or IP address of the MQTT broker.                      |
| **MQTT_BROKER_PORT**           | `None`                   | Port of the MQTT broker.                                        |
| **MQTT_BROKER_USERNAME**       | `None`                   | Username for authenticating with the MQTT broker (if required). |
| **MQTT_BROKER_PASSWORD**       | `None`                   | Password for authenticating with the MQTT broker (if required). |
| **MQTT_TOPIC_FORECAST**        | `tirex/forecast/request` | MQTT topic to subscribe to for receiving forecast requests.     |
| **MQTT_TOPIC_FORECAST_RESULT** | `tirex/forecast/result`  | MQTT topic to publish successful forecast results to.           |
| **MQTT_TOPIC_FORECAST_ERROR**  | `tirex/forecast/error`   | MQTT topic to publish forecast error messages to.               |


## Build and run the docker container

### CPU Container

Build the CPU image:
```
docker build -f Dockerfile.cpu -t tirex-inference-cpu .
```

Run the CPU container:
```
docker run -p 8000:8000 -it tirex-inference-cpu
```

### GPU Container

Build the GPU Docker image:
```
docker build -f Dockerfile.gpu -t tirex-inference-gpu .
```

Run the GPU container:
```
docker run --gpus all -p 8000:8000 -it tirex-inference-gpu
```

## Development Setup

### Install Python dependencies:
```
pip install -r requirements.txt -r requirements-dev.txt
```

### Run the server:
```
python -m app.main
```

### Run Tests:
Run while starting the server locally:
```
pytest tests
```

Run tests against a running container:
```
TEST_START_SERVER=0 TEST_PORT=8000 pytest tests -s
```

## License

TiRex is licensed under the [NXAI community license](../LICENSE).
