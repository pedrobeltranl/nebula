import os
import sys
import json
import logging
import asyncio

# Setup environment variables
ROOT_DIR = "/Users/pedrobeltranlopez/Desktop/NEBULA2/nebula"
os.environ["NEBULA_ROOT_HOST"] = ROOT_DIR
os.environ["NEBULA_HOST_PLATFORM"] = "mac"
os.environ["NEBULA_CONFIG_DIR"] = os.path.join(ROOT_DIR, "app/config")
os.environ["NEBULA_LOGS_DIR"] = os.path.join(ROOT_DIR, "app/logs")
os.environ["NEBULA_CERTS_DIR"] = os.path.join(ROOT_DIR, "app/certs")
os.environ["NEBULA_CONTROLLER_HOST"] = "localhost"
os.environ["NEBULA_CONTROLLER_PORT"] = "5050"
os.environ["NEBULA_CONTROLLER_NAME"] = "controller"

sys.path.append(ROOT_DIR)

from nebula.controller.scenarios import ScenarioManagement

# Load existing scenario.json
SCENARIO_PATH = os.path.join(ROOT_DIR, "app/logs/nebula_DFL_2026_02_19_22_41_27 2/scenario.json")

def main():
    logging.basicConfig(level=logging.INFO)
    try:
        with open(SCENARIO_PATH, "r") as f:
            scenario_data = json.load(f)

        # Modify for LOCAL PROCESS execution
        scenario_data["deployment"] = "process"
        scenario_data["rounds"] = 30
        scenario_data["scenario_title"] = "verification_run_sybil_fix"

        # Patch IPs to LOCALHOST
        if "nodes" in scenario_data:
            for node_id, node_info in scenario_data["nodes"].items():
                node_info["ip"] = "127.0.0.1"
                # Ports are already unique in this scenario (45001-45010)

        # Ensure Honeypot is enabled
        if "honeypot" not in scenario_data:
            scenario_data["honeypot"] = {"enabled": True, "mode": "fixed", "seed": 0.5}
        scenario_data["honeypot"]["enabled"] = True

        print("Generating Verification Scenario Config...")
        sm = ScenarioManagement(scenario_data, user="verification_agent")
        print(f"Scenario Name: {sm.scenario_name}")

        async def run_setup():
            await sm.load_configurations_and_start_nodes()
            print(f"PROCESS START SCRIPT GENERATED AT: {sm.config_dir}/current_scenario_commands.sh")
            print(f"LOGS DIR: {sm.log_dir}/{sm.scenario_name}")

        asyncio.run(run_setup())

    except Exception as e:
        logging.exception("Failed to launch scenario")
        print(e)

    except Exception as e:
        logging.exception("Failed to launch scenario")
        print(e)

    # Async wrapper not needed unless we call async methods
    # We will redefine main logic below to replace this block


if __name__ == "__main__":
    main()
