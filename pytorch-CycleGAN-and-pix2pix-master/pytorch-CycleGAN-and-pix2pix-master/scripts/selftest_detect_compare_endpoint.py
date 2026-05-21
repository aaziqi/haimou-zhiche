import json
import threading
import time
from http.server import ThreadingHTTPServer
from urllib.request import Request, urlopen


def post_json(url: str, body: dict, timeout: int = 60):
    req = Request(url, data=json.dumps(body).encode("utf-8"), headers={"Content-Type": "application/json"})
    with urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode("utf-8"))


def main():
    import interactive_demo_server as srv

    srv.ensure_runtime_root()
    server = ThreadingHTTPServer(("127.0.0.1", 0), srv.make_handler(gpu_ids="-1"))
    port = int(server.server_address[1])
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    time.sleep(0.4)

    body = {
        "raw_image_url": "/runtime/enhance_jobs/enhance_59920ea19f/result/improved_mpcgan/assets/input/0_input.png",
        "enhanced_image_url": "/runtime/enhance_jobs/enhance_59920ea19f/result/improved_mpcgan/assets/output/0_input_fake.png",
        "detector_key": "default_detector",
        "conf": 0.25,
        "imgsz": 640,
    }
    payload = post_json(f"http://127.0.0.1:{port}/api/detect_compare", body, timeout=120)
    assert payload.get("ok") is True, payload
    assert payload.get("compare_vis_url"), payload
    assert payload.get("raw_json_url"), payload
    assert payload.get("enhanced_json_url"), payload
    compare_path = srv.web_url_to_path(payload["compare_vis_url"])
    raw_json_path = srv.web_url_to_path(payload["raw_json_url"])
    enh_json_path = srv.web_url_to_path(payload["enhanced_json_url"])
    assert compare_path.exists(), compare_path
    assert raw_json_path.exists(), raw_json_path
    assert enh_json_path.exists(), enh_json_path

    server.shutdown()
    server.server_close()
    print("ok", payload.get("job_id"), payload.get("compare_vis_url"))


if __name__ == "__main__":
    main()

