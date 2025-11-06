# Copyright (c) NXAI GmbH.
# This software may be used and distributed according to the terms of the NXAI Community License Agreement.

import atexit
import multiprocessing as mp

from app.config import Settings


def run_http():
    import uvicorn

    from app.http_server import app

    settings = Settings()
    uvicorn.run(app, host=settings.http_host, port=settings.http_port)


def run_mqtt():
    from app.mqtt_server import TirexMQTTClient

    settings = Settings()
    client = TirexMQTTClient(settings)
    client.connect()


def main():
    mp.set_start_method("spawn", force=True)
    config = Settings()
    processes: list[mp.Process] = []

    processes.append(mp.Process(target=run_http, name="HTTP Server"))

    if config.mqtt_enabled == 1:
        processes.append(mp.Process(target=run_mqtt, name="MQTT Server"))

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
