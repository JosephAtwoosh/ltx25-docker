#!/usr/bin/env node
/**
 * Minimal Vast.ai API client.
 *
 * Why this exists: this machine's Windows schannel stack cannot do TLS inside the
 * sandbox (SEC_E_NO_CREDENTIALS), which breaks PowerShell, curl and git. Node ships
 * its own OpenSSL, so HTTPS works here. No dependencies.
 *
 * Auth: --key-file <path>, env VAST_API_KEY, or ~/.vastai/vastai_api_key.
 * Usage:
 *   node vast.js whoami
 *   node vast.js instances
 *   node vast.js search '[RTX 4090,RTX 5090]' 31000 120
 *   node vast.js destroy <instance_id>        # only with --yes
 */
"use strict";

const https = require("https");
const fs = require("fs");
const os = require("os");
const path = require("path");

const HOST = "console.vast.ai";

function findKey() {
  if (process.env.VAST_API_KEY) return { key: process.env.VAST_API_KEY, from: "env VAST_API_KEY" };
  const candidates = [
    path.join(os.homedir(), ".vastai", "vastai_api_key"),
    path.join(os.homedir(), ".config", "vastai", "vast_api_key"),
    path.join(process.env.APPDATA || "", "vastai", "vast_api_key"),
  ];
  for (const p of candidates) {
    try {
      const key = fs.readFileSync(p, "utf8").trim();
      if (key) return { key, from: p };
    } catch {
      /* try next */
    }
  }
  return null;
}

function request(method, apiPath, body, key) {
  return new Promise((resolve, reject) => {
    const data = body === undefined ? null : JSON.stringify(body);
    const headers = { Accept: "application/json" };
    if (key) headers.Authorization = `Bearer ${key}`;
    if (data) {
      headers["Content-Type"] = "application/json";
      headers["Content-Length"] = Buffer.byteLength(data);
    }
    const req = https.request(
      { hostname: HOST, path: apiPath, method, headers, timeout: 40000 },
      (res) => {
        let buf = "";
        res.on("data", (c) => (buf += c));
        res.on("end", () => {
          let parsed = buf;
          try {
            parsed = JSON.parse(buf);
          } catch {
            /* leave as text */
          }
          resolve({ status: res.statusCode, body: parsed });
        });
      },
    );
    req.on("error", reject);
    req.on("timeout", () => req.destroy(new Error("request timed out")));
    if (data) req.write(data);
    req.end();
  });
}

async function main() {
  const [cmd, ...args] = process.argv.slice(2);
  const found = findKey();

  const needKey = () => {
    if (!found) {
      console.error(
        "No API key found.\n" +
          "  Set VAST_API_KEY, or run `vastai set api-key <key>`,\n" +
          "  or write the key to ~/.vastai/vastai_api_key",
      );
      process.exit(2);
    }
    return found.key;
  };

  switch (cmd) {
    case "whoami": {
      const key = needKey();
      const r = await request("GET", "/api/v0/users/current/", undefined, key);
      console.log("key source:", found.from);
      console.log("status:", r.status);
      console.log(JSON.stringify(r.body, null, 2).slice(0, 1200));
      break;
    }

    case "instances": {
      // Read-only and free: tells us whether anything is already running before
      // spending money on a new rental.
      const key = needKey();
      const r = await request("GET", "/api/v0/instances/", undefined, key);
      console.log("key source:", found.from);
      console.log("status:", r.status);
      const list = (r.body && r.body.instances) || r.body;
      console.log(JSON.stringify(list, null, 2).slice(0, 4000));
      break;
    }

    case "search": {
      // usage: search 'RTX 4090,RTX 5090' <min_vram_mb> <min_disk_gb>
      const names = (args[0] || "RTX 4090,RTX 5090").split(",").map((s) => s.trim());
      const minVram = Number(args[1] || 31000);
      const minDisk = Number(args[2] || 120);
      const q = {
        gpu_name: { in: names },
        num_gpus: { gte: 1 },
        gpu_ram: { gte: minVram },
        disk_space: { gte: minDisk },
        rentable: { eq: true },
        verified: { eq: true },
        type: "ondemand",
        order: [["dph_total", "asc"]],
        limit: 10,
      };
      const r = await request("POST", "/api/v0/bundles/", q, found ? found.key : null);
      if (r.status !== 200) {
        console.error("status", r.status, JSON.stringify(r.body).slice(0, 500));
        process.exit(1);
      }
      const rows = (r.body.offers || []).map((o) => ({
        id: o.id,
        gpu: `${o.num_gpus}x ${o.gpu_name}`,
        vram_gb: Math.round(o.gpu_total_ram / 1024),
        cuda: o.cuda_max_good,
        driver: o.driver_version,
        ram_gb: Math.round(o.cpu_ram / 1024),
        disk_gb: Math.round(o.disk_space),
        usd_hr: Number(o.dph_total.toFixed(3)),
        geo: o.geolocation,
      }));
      console.log(`Found ${rows.length} offers (VRAM >= ${minVram} MB, disk >= ${minDisk} GB):`);
      console.table(rows);
      console.log("\nNOTE: cuda_max_good must be >= 13.2 for the LTX-2.5 image.");
      break;
    }

    case "destroy": {
      const key = needKey();
      const id = args[0];
      if (!id || !args.includes("--yes")) {
        console.error("Refusing to destroy without an instance id and --yes");
        process.exit(2);
      }
      const r = await request("DELETE", `/api/v0/instances/${id}/`, undefined, key);
      console.log("status:", r.status, JSON.stringify(r.body).slice(0, 500));
      break;
    }

    default:
      console.error(
        "usage: node vast.js <whoami|instances|search|destroy> [...]\n" +
          "  whoami                     verify the key\n" +
          "  instances                  list this account's instances (free)\n" +
          '  search \'["RTX 4090"]\' 31000 120\n' +
          "  destroy <id> --yes",
      );
      process.exit(1);
  }
}

main().catch((e) => {
  console.error("error:", e.message);
  process.exit(1);
});
