import json

from minestudio.simulator.minerl.env import database_manager


def main() -> None:
    with database_manager.resetting_lock:
        resetting_list = list(database_manager.database["resetting_list"])
        for request_uuid in resetting_list:
            key = (request_uuid, "resetting_request_time")
            if key in database_manager.database:
                del database_manager.database[key]
        database_manager.database["resetting_list"] = []
    print(json.dumps({"cleared_reset_slots": len(resetting_list)}))


if __name__ == "__main__":
    main()
