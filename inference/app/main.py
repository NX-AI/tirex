# Copyright (c) NXAI GmbH.
# This software may be used and distributed according to the terms of the NXAI Community License Agreement.

import atexit
import multiprocessing as mp

from app.config import Settings


def run_http(config: Settings):
    import uvicorn

    from app.http_server import app

    uvicorn.run(app, host=config.http_host, port=config.http_port)


def run_mqtt(config: Settings):
    from app.mqtt_server import TirexMQTTClient

    client = TirexMQTTClient(config)
    client.connect()


def main():
    mp.set_start_method("spawn", force=True)
    config = Settings()
    processes: list[mp.Process] = []

    processes.append(mp.Process(target=run_http, args=(config,), name="HTTP Server"))

    if config.mqtt_enabled == 1:
        processes.append(mp.Process(target=run_mqtt, args=(config,), name="MQTT Server"))

    for p in processes:
        p.start()

    for p in processes:
        p.join()

    def stop_processes():
        for p in processes:
            p.kill()

    atexit.register(stop_processes)


if __name__ == "__main__":
    main()
