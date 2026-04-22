#!/usr/bin/env python3

from __future__ import annotations

import argparse
import functools
import http.server
import socketserver


class NoCacheRequestHandler(http.server.SimpleHTTPRequestHandler):
    def end_headers(self) -> None:
        self.send_header("Cache-Control", "no-store, max-age=0, must-revalidate")
        self.send_header("Pragma", "no-cache")
        self.send_header("Expires", "0")
        super().end_headers()


class ReusableTCPServer(socketserver.TCPServer):
    allow_reuse_address = True


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=4173)
    parser.add_argument("directory")
    args = parser.parse_args()

    handler = functools.partial(NoCacheRequestHandler, directory=args.directory)
    with ReusableTCPServer(("127.0.0.1", args.port), handler) as httpd:
        print(f"Serving {args.directory} on http://127.0.0.1:{args.port}")
        httpd.serve_forever()


if __name__ == "__main__":
    main()
