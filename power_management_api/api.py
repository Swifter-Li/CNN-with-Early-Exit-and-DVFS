import time
from multiprocessing import Process, Value
import subprocess
from typing import List, Dict

_local_dict = {}


def _trace_power_log(period: float, path: str, stop: Value) -> int:
    """
    This is a private function and shouldn't be called from outside.

    :param float period: period to sleep
    :param str path: path to log output file
    :param Value stop: value shared by parent and child processes
    :return: 0 on success, negative numbers on errors
    """
    power_trace_path = "/sys/bus/i2c/drivers/ina3221x/0-0041/iio:device1/in_power0_input"
    log = open(path, "a")
    while stop.value != 1:
        power_trace = open(power_trace_path, "r")
        power = power_trace.read()
        power_trace.close()
        log.write("ms: {0}, power: {1}".format(int(time.time() * 1000), power))
        time.sleep(period)
    log.close()
    return 0


def start_power_log(period: float, path: str) -> int:
    """
    This function should be called with super-user privilege.
    This function **shouldn't** be called when a recording has already been started.

    It starts a child process recording input power to the SoC in specified period and
    outputting the result to a specified file.

    :param float period: period to sleep
    :param str path: **absolute** path to log output file
    :return: 0 on success, negative numbers on errors
    """
    if "power_log_subprocess" in _local_dict.keys():
        return -1
    stop = Value("i", 0)
    _local_dict["power_log_stop"] = stop
    p = Process(target=_trace_power_log, args=(period, path, stop))
    _local_dict["power_log_subprocess"] = p
    p.start()
    return 0


def stop_power_log() -> int:
    """
    This function **shouldn't** be called when there isn't a recording.
    Warning: Only call it when time is not critical, since it is blocking
    and will wait the subprocess to finish execution, which at most costs
    ``period`` of time.

    It stops the child process recording power.

    :return: 0 on success, negative numbers on errors
    """
    if "power_log_subprocess" not in _local_dict.keys():
        return -1
    _local_dict["power_log_stop"].value = 1
    _local_dict["power_log_subprocess"].join()
    _local_dict.pop("power_log_stop")
    _local_dict.pop("power_log_subprocess")
    return 0


def sleep_with_timer(second: int) -> int:
    """
    This function should be called with super-user privilege.
    Both setting up the timer and suspending the system need
    the privilege to function correctly.

    :param int second: Number of seconds you want to sleep for
    :return: 0 on success
    :rtype: int
    """
    if second < 0:
        return -1
    rtc_path = "/sys/class/rtc/rtc0/wakealarm"
    curr_t = int(time.time())
    timer = open(rtc_path, "w")
    timer.write(str(curr_t + second))
    timer.close()
    sleep = subprocess.call(["systemctl", "suspend"])
    if sleep != 0:
        return sleep
    return 0


def get_cpu_info(cpu_idx=None, cpu_type=True, cpu_online=True, min_freq=True, max_freq=True, cur_freq=True,
                 available_freq=True, cur_gov=True, available_gov=True) -> List[Dict]:
    """
    With no arguments, this function reads a series of virtual files
    and return a complete list of all possible info. If it receives arguments,
    it will only return corresponding values.
    CPU index other than 0 to 5 will be ignored.

    :param List[int] cpu_idx: Index of CPUs whose info is needed
    :param bool cpu_type: Return cpu type or not
    :param bool cpu_online: Return cpu online state or not
    :param bool min_freq: Return minimum frequency or not
    :param bool max_freq: Return maximum frequency or not
    :param bool cur_freq: Return current frequency or not
    :param bool available_freq: Return available frequencies or not
    :param bool cur_gov: Return current governor or not
    :param bool available_gov: Return available governors or not
    :return: A list of dictionaries where cpu information is stored
    """
    available_idx = (0, 1, 2, 3, 4, 5)
    cpu_path = "/sys/devices/system/cpu/cpu{0}"

    if cpu_idx is None:
        cpu_idx = [0, 1, 2, 3, 4, 5]
    result = []
    for i in cpu_idx:
        if i not in available_idx:
            continue
        cpu_info = {"cpu_idx": i}
        if cpu_type:
            cpu_info["cpu_type"] = "Denver" if i == 1 or i == 2 else "A57"
        if cpu_online:
            f = open(cpu_path.format(i) + "/online", "r")
            content = f.read()
            f.close()
            cpu_info["cpu_online"] = True if content.rstrip() == "1" else False
        if min_freq:
            f = open(cpu_path.format(i) + "/cpufreq/scaling_min_freq", "r")
            content = f.read()
            f.close()
            cpu_info["min_freq"] = int(content.rstrip())
        if max_freq:
            f = open(cpu_path.format(i) + "/cpufreq/scaling_max_freq", "r")
            content = f.read()
            f.close()
            cpu_info["max_freq"] = int(content.rstrip())
        if cur_freq:
            f = open(cpu_path.format(i) + "/cpufreq/cpuinfo_cur_freq", "r")
            content = f.read()
            f.close()
            if content.rstrip() == "<unknown>":
                cpu_info["cur_freq"] = 0
            else:
                cpu_info["cur_freq"] = int(content.rstrip())
        if available_freq:
            online_f = open(cpu_path.format(i) + "/online", "r")
            online_content = online_f.read()
            online_f.close()
            if online_content.rstrip() != "1":
                cpu_info["available_freq"] = []
            else:
                f = open(cpu_path.format(i) + "/cpufreq/scaling_available_frequencies", "r")
                content = f.read()
                f.close()
                cpu_info["available_freq"] = [int(val) for val in content.rstrip().split()]
        if cur_gov:
            f = open(cpu_path.format(i) + "/cpufreq/scaling_governor", "r")
            content = f.read()
            f.close()
            cpu_info["cur_gov"] = content.rstrip()
        if available_gov:
            f = open(cpu_path.format(i) + "/cpufreq/scaling_available_governors", "r")
            content = f.read()
            f.close()
            cpu_info["available_gov"] = content.rstrip().split()
        result.append(cpu_info)
    return result


def set_cpu_state(cpu_targets: List[Dict]) -> int:
    """
    This function takes in a list of dicts and sets the cpu parameters accordingly.
    For each dict, "cpu_idx" is a necessary field and the possible four options are
    "cpu_online", "min_freq", "max_freq", and "governor".

    If the "cpu_online" field is False, or "cpu_online" is not specified and cpu is offline,
    the frequency fields and governor field will be ignored.

    There are many possible errors and here's a list error code:

    - -1: "cpu_idx" field is missing.
    - -2: The last online cpu is being offline.
    - -3: The specified minimum frequency is not in available frequency list.
    - -4: The specified maximum frequency is not in available frequency list.
    - -5: Only minimum frequency is specified, and it is greater than current maximum frequency.
    - -6: Only maximum frequency is specified, and it is smaller than current minimum frequency.
    - -7: The specified maximum frequency is smaller than the specified minimum frequency.
    - -8: The specified governor is not in available governor list.

    If multiple dicts in the list have errors, the error code will be concatenated together.
    For example, if the first dict doesn't specify "cpu_idx", and the third dict gives a
    wrong minimum frequency, the return value will be -103. If only the third dict gives a
    wrong minimum frequency, the return value will be -3.

    :param List[Dict] cpu_targets: A list of dicts which contain cpu parameters.
    :return: Refer to detailed doc.
    """
    cpu_path = "/sys/devices/system/cpu/cpu{0}"
    error_code = 0

    cpu_info = get_cpu_info()
    cpu_info_dict = dict()
    for info in cpu_info:
        cpu_info_dict[info["cpu_idx"]] = info
    for cpu_target in cpu_targets:
        idx_flag = "cpu_idx" in cpu_target
        online_flag = "cpu_online" in cpu_target
        min_flag = "min_freq" in cpu_target
        max_flag = "max_freq" in cpu_target
        gov_flag = "governor" in cpu_target
        if not idx_flag:
            error_code *= 10
            error_code += 1
            continue
        if online_flag and not cpu_target["cpu_online"]:
            if not cpu_info_dict[cpu_target["cpu_idx"]]["cpu_online"]:
                continue
            if len([cpu for cpu in cpu_info if cpu["cpu_online"]]) == 1:
                error_code *= 10
                error_code += 2
                continue
            f = open(cpu_path.format(cpu_target["cpu_idx"]) + "/online", "w")
            f.write("0\n")
            f.close()
            cpu_info_dict[cpu_target["cpu_idx"]]["cpu_online"] = False
            continue
        if online_flag and cpu_target["cpu_online"] and not cpu_info_dict[cpu_target["cpu_idx"]]["cpu_online"]:
            f = open(cpu_path.format(cpu_target["cpu_idx"]) + "/online", "w")
            f.write("1\n")
            f.close()
        if not online_flag and not cpu_info_dict[cpu_target["cpu_idx"]]["cpu_online"]:
            continue
        if len(cpu_info_dict[cpu_target["cpu_idx"]]["available_freq"]) == 0:
            cpu_info_dict[cpu_target["cpu_idx"]] = get_cpu_info([cpu_target["cpu_idx"]])[0]
        if min_flag and cpu_target["min_freq"] not in cpu_info_dict[cpu_target["cpu_idx"]]["available_freq"]:
            error_code *= 10
            error_code += 3
            continue
        if max_flag and cpu_target["max_freq"] not in cpu_info_dict[cpu_target["cpu_idx"]]["available_freq"]:
            error_code *= 10
            error_code += 4
            continue
        if min_flag and not max_flag and cpu_target["min_freq"] > cpu_info_dict[cpu_target["cpu_idx"]]["max_freq"]:
            error_code *= 10
            error_code += 5
            continue
        if max_flag and not min_flag and cpu_target["max_freq"] < cpu_info_dict[cpu_target["cpu_idx"]]["min_freq"]:
            error_code *= 10
            error_code += 6
            continue
        if min_flag and max_flag and cpu_target["min_freq"] > cpu_target["max_freq"]:
            error_code *= 10
            error_code += 7
            continue
        if min_flag and not max_flag:
            f = open(cpu_path.format(cpu_target["cpu_idx"]) + "/cpufreq/scaling_min_freq", "w")
            f.write(str(cpu_target["min_freq"]))
            f.close()
        if max_flag and not min_flag:
            f = open(cpu_path.format(cpu_target["cpu_idx"]) + "/cpufreq/scaling_max_freq", "w")
            f.write(str(cpu_target["max_freq"]))
            f.close()
        if min_flag and max_flag:
            if cpu_target["min_freq"] > cpu_info_dict[cpu_target["cpu_idx"]]["max_freq"]:
                f_max = open(cpu_path.format(cpu_target["cpu_idx"]) + "/cpufreq/scaling_max_freq", "w")
                f_max.write(str(cpu_target["max_freq"]))
                f_max.close()
                f_min = open(cpu_path.format(cpu_target["cpu_idx"]) + "/cpufreq/scaling_min_freq", "w")
                f_min.write(str(cpu_target["min_freq"]))
                f_min.close()
            else:
                f_min = open(cpu_path.format(cpu_target["cpu_idx"]) + "/cpufreq/scaling_min_freq", "w")
                f_min.write(str(cpu_target["min_freq"]))
                f_min.close()
                f_max = open(cpu_path.format(cpu_target["cpu_idx"]) + "/cpufreq/scaling_max_freq", "w")
                f_max.write(str(cpu_target["max_freq"]))
                f_max.close()
        if gov_flag:
            if cpu_target["governor"] not in cpu_info_dict[cpu_target["cpu_idx"]]["available_gov"]:
                error_code *= 10
                error_code += 8
                continue
            f = open(cpu_path.format(cpu_target["cpu_idx"]) + "/cpufreq/scaling_governor", "w")
            f.write(cpu_target["governor"])
            f.close()
        error_code *= 10
    if error_code == 0:
        return 0
    return -error_code


def get_gpu_info(min_freq=True, max_freq=True, cur_freq=True, available_freq=True, cur_gov=True,
                 available_gov=True) -> Dict:
    """
    With no arguments, this function reads a series of virtual files
    and return all possible info. If it receives arguments, it will only
    return corresponding values.

    :param bool min_freq: Return minimum frequency or not.
    :param bool max_freq: Return maximum frequency or not.
    :param bool cur_freq: Return current frequency or not.
    :param bool available_freq: Return available frequencies or not.
    :param bool cur_gov: Return current governor or not.
    :param bool available_gov: Return available governors or not.
    :return: A dictionary where gpu information is stored.
    """
    gpu_path = "/sys/devices/gpu.0/devfreq/17000000.gp10b"
    gpu_info = {}
    if min_freq:
        f = open(gpu_path + "/min_freq", "r")
        content = f.read()
        f.close()
        gpu_info["min_freq"] = int(content.rstrip())
    if max_freq:
        f = open(gpu_path + "/max_freq", "r")
        content = f.read()
        f.close()
        gpu_info["max_freq"] = int(content.rstrip())
    if cur_freq:
        f = open(gpu_path + "/cur_freq", "r")
        content = f.read()
        f.close()
        gpu_info["cur_freq"] = int(content.rstrip())
    if available_freq:
        f = open(gpu_path + "/available_frequencies", "r")
        content = f.read()
        f.close()
        gpu_info["available_freq"] = [int(val) for val in content.rstrip().split()]
    if cur_gov:
        f = open(gpu_path + "/governor", "r")
        content = f.read()
        f.close()
        gpu_info["cur_gov"] = content.rstrip()
    if available_gov:
        f = open(gpu_path + "/available_governors", "r")
        content = f.read()
        f.close()
        gpu_info["available_gov"] = content.rstrip().split()
    return gpu_info


def set_gpu_state(gpu_target: Dict) -> int:
    """
    **Note: Governor cannot be set currently. There may be some errors, and the code is commented.**

    This function takes in a dict and sets the gpu parameters accordingly.
    The possible three options are "min_freq", "max_freq", and "governor".

    There are many possible errors and here's a list error code:

    - -1: The specified minimum frequency is not in available frequency list.
    - -2: The specified maximum frequency is not in available frequency list.
    - -3: Only minimum frequency is specified, and it is greater than current maximum frequency.
    - -4: Only maximum frequency is specified, and it is smaller than current minimum frequency.
    - -5: The specified maximum frequency is smaller than the specified minimum frequency.
    - -6: The specified governor is not in available governor list.

    :param Dict gpu_target: A dict which contains gpu parameters.
    :return: Refer to detailed doc.
    """
    gpu_path = "/sys/devices/gpu.0/devfreq/17000000.gp10b"

    gpu_info = get_gpu_info()
    min_flag = "min_freq" in gpu_target
    max_flag = "max_freq" in gpu_target
    gov_flag = "governor" in gpu_target
    if min_flag and gpu_target["min_freq"] not in gpu_info["available_freq"]:
        return -1
    if max_flag and gpu_target["max_freq"] not in gpu_info["available_freq"]:
        return -2
    if min_flag and not max_flag and gpu_target["min_freq"] > gpu_info["max_freq"]:
        return -3
    if max_flag and not min_flag and gpu_target["max_freq"] < gpu_info["min_freq"]:
        return -4
    if min_flag and max_flag and gpu_target["min_freq"] > gpu_target["max_freq"]:
        return -5
    if min_flag and not max_flag:
        f = open(gpu_path + "/min_freq", "w")
        f.write(str(gpu_target["min_freq"]))
        f.close()
    if max_flag and not min_flag:
        f = open(gpu_path + "/max_freq", "w")
        f.write(str(gpu_target["max_freq"]))
        f.close()
    if min_flag and max_flag:
        if gpu_target["min_freq"] > gpu_info["max_freq"]:
            f_max = open(gpu_path + "/max_freq", "w")
            f_max.write(str(gpu_target["max_freq"]))
            f_max.close()
            f_min = open(gpu_path + "/min_freq", "w")
            f_min.write(str(gpu_target["min_freq"]))
            f_min.close()
        else:
            f_min = open(gpu_path + "/min_freq", "w")
            f_min.write(str(gpu_target["min_freq"]))
            f_min.close()
            f_max = open(gpu_path + "/max_freq", "w")
            f_max.write(str(gpu_target["max_freq"]))
            f_max.close()
    if gov_flag:
        if gpu_target["governor"] not in gpu_info["available_gov"]:
            return -6
        # f = open(gpu_path + "/governor", "w")
        # f.write(gpu_target["governor"])
        # f.close()
    return 0
import time
from multiprocessing import Process, Value
import subprocess
from typing import List, Dict

_local_dict = {}


def _trace_power_log(period: float, path: str, stop: Value) -> int:
    """
    This is a private function and shouldn't be called from outside.

    :param float period: period to sleep
    :param str path: path to log output file
    :param Value stop: value shared by parent and child processes
    :return: 0 on success, negative numbers on errors
    """
    power_trace_path = "/sys/bus/i2c/drivers/ina3221x/0-0041/iio:device1/in_power0_input"
    log = open(path, "a")
    while stop.value != 1:
        power_trace = open(power_trace_path, "r")
        power = power_trace.read()
        power_trace.close()
        log.write("ms: {0}, power: {1}".format(int(time.time() * 1000), power))
        time.sleep(period)
    log.close()
    return 0


def start_power_log(period: float, path: str) -> int:
    """
    This function should be called with super-user privilege.
    This function **shouldn't** be called when a recording has already been started.

    It starts a child process recording input power to the SoC in specified period and
    outputting the result to a specified file.

    :param float period: period to sleep
    :param str path: **absolute** path to log output file
    :return: 0 on success, negative numbers on errors
    """
    if "power_log_subprocess" in _local_dict.keys():
        return -1
    stop = Value("i", 0)
    _local_dict["power_log_stop"] = stop
    p = Process(target=_trace_power_log, args=(period, path, stop))
    _local_dict["power_log_subprocess"] = p
    p.start()
    return 0


def stop_power_log() -> int:
    """
    This function **shouldn't** be called when there isn't a recording.
    Warning: Only call it when time is not critical, since it is blocking
    and will wait the subprocess to finish execution, which at most costs
    ``period`` of time.

    It stops the child process recording power.

    :return: 0 on success, negative numbers on errors
    """
    if "power_log_subprocess" not in _local_dict.keys():
        return -1
    _local_dict["power_log_stop"].value = 1
    _local_dict["power_log_subprocess"].join()
    _local_dict.pop("power_log_stop")
    _local_dict.pop("power_log_subprocess")
    return 0


def sleep_with_timer(second: int) -> int:
    """
    This function should be called with super-user privilege.
    Both setting up the timer and suspending the system need
    the privilege to function correctly.

    :param int second: Number of seconds you want to sleep for
    :return: 0 on success
    :rtype: int
    """
    if second < 0:
        return -1
    rtc_path = "/sys/class/rtc/rtc0/wakealarm"
    curr_t = int(time.time())
    timer = open(rtc_path, "w")
    timer.write(str(curr_t + second))
    timer.close()
    sleep = subprocess.call(["systemctl", "suspend"])
    if sleep != 0:
        return sleep
    return 0


def get_cpu_info(cpu_idx=None, cpu_type=True, cpu_online=True, min_freq=True, max_freq=True, cur_freq=True,
                 available_freq=True, cur_gov=True, available_gov=True) -> List[Dict]:
    """
    With no arguments, this function reads a series of virtual files
    and return a complete list of all possible info. If it receives arguments,
    it will only return corresponding values.
    CPU index other than 0 to 5 will be ignored.

    :param List[int] cpu_idx: Index of CPUs whose info is needed
    :param bool cpu_type: Return cpu type or not
    :param bool cpu_online: Return cpu online state or not
    :param bool min_freq: Return minimum frequency or not
    :param bool max_freq: Return maximum frequency or not
    :param bool cur_freq: Return current frequency or not
    :param bool available_freq: Return available frequencies or not
    :param bool cur_gov: Return current governor or not
    :param bool available_gov: Return available governors or not
    :return: A list of dictionaries where cpu information is stored
    """
    available_idx = (0, 1, 2, 3, 4, 5)
    cpu_path = "/sys/devices/system/cpu/cpu{0}"

    if cpu_idx is None:
        cpu_idx = [0, 1, 2, 3, 4, 5]
    result = []
    for i in cpu_idx:
        if i not in available_idx:
            continue
        cpu_info = {"cpu_idx": i}
        if cpu_type:
            cpu_info["cpu_type"] = "Denver" if i == 1 or i == 2 else "A57"
        if cpu_online:
            f = open(cpu_path.format(i) + "/online", "r")
            content = f.read()
            f.close()
            cpu_info["cpu_online"] = True if content.rstrip() == "1" else False
        if min_freq:
            f = open(cpu_path.format(i) + "/cpufreq/scaling_min_freq", "r")
            content = f.read()
            f.close()
            cpu_info["min_freq"] = int(content.rstrip())
        if max_freq:
            f = open(cpu_path.format(i) + "/cpufreq/scaling_max_freq", "r")
            content = f.read()
            f.close()
            cpu_info["max_freq"] = int(content.rstrip())
        if cur_freq:
            f = open(cpu_path.format(i) + "/cpufreq/cpuinfo_cur_freq", "r")
            content = f.read()
            f.close()
            if content.rstrip() == "<unknown>":
                cpu_info["cur_freq"] = 0
            else:
                cpu_info["cur_freq"] = int(content.rstrip())
        if available_freq:
            online_f = open(cpu_path.format(i) + "/online", "r")
            online_content = online_f.read()
            online_f.close()
            if online_content.rstrip() != "1":
                cpu_info["available_freq"] = []
            else:
                f = open(cpu_path.format(i) + "/cpufreq/scaling_available_frequencies", "r")
                content = f.read()
                f.close()
                cpu_info["available_freq"] = [int(val) for val in content.rstrip().split()]
        if cur_gov:
            f = open(cpu_path.format(i) + "/cpufreq/scaling_governor", "r")
            content = f.read()
            f.close()
            cpu_info["cur_gov"] = content.rstrip()
        if available_gov:
            f = open(cpu_path.format(i) + "/cpufreq/scaling_available_governors", "r")
            content = f.read()
            f.close()
            cpu_info["available_gov"] = content.rstrip().split()
        result.append(cpu_info)
    return result


def set_cpu_state(cpu_targets: List[Dict]) -> int:
    """
    This function takes in a list of dicts and sets the cpu parameters accordingly.
    For each dict, "cpu_idx" is a necessary field and the possible four options are
    "cpu_online", "min_freq", "max_freq", and "governor".

    If the "cpu_online" field is False, or "cpu_online" is not specified and cpu is offline,
    the frequency fields and governor field will be ignored.

    There are many possible errors and here's a list error code:

    - -1: "cpu_idx" field is missing.
    - -2: The last online cpu is being offline.
    - -3: The specified minimum frequency is not in available frequency list.
    - -4: The specified maximum frequency is not in available frequency list.
    - -5: Only minimum frequency is specified, and it is greater than current maximum frequency.
    - -6: Only maximum frequency is specified, and it is smaller than current minimum frequency.
    - -7: The specified maximum frequency is smaller than the specified minimum frequency.
    - -8: The specified governor is not in available governor list.

    If multiple dicts in the list have errors, the error code will be concatenated together.
    For example, if the first dict doesn't specify "cpu_idx", and the third dict gives a
    wrong minimum frequency, the return value will be -103. If only the third dict gives a
    wrong minimum frequency, the return value will be -3.

    :param List[Dict] cpu_targets: A list of dicts which contain cpu parameters.
    :return: Refer to detailed doc.
    """
    cpu_path = "/sys/devices/system/cpu/cpu{0}"
    error_code = 0

    cpu_info = get_cpu_info()
    cpu_info_dict = dict()
    for info in cpu_info:
        cpu_info_dict[info["cpu_idx"]] = info
    for cpu_target in cpu_targets:
        idx_flag = "cpu_idx" in cpu_target
        online_flag = "cpu_online" in cpu_target
        min_flag = "min_freq" in cpu_target
        max_flag = "max_freq" in cpu_target
        gov_flag = "governor" in cpu_target
        if not idx_flag:
            error_code *= 10
            error_code += 1
            continue
        if online_flag and not cpu_target["cpu_online"]:
            if not cpu_info_dict[cpu_target["cpu_idx"]]["cpu_online"]:
                continue
            if len([cpu for cpu in cpu_info if cpu["cpu_online"]]) == 1:
                error_code *= 10
                error_code += 2
                continue
            f = open(cpu_path.format(cpu_target["cpu_idx"]) + "/online", "w")
            f.write("0\n")
            f.close()
            cpu_info_dict[cpu_target["cpu_idx"]]["cpu_online"] = False
            continue
        if online_flag and cpu_target["cpu_online"] and not cpu_info_dict[cpu_target["cpu_idx"]]["cpu_online"]:
            f = open(cpu_path.format(cpu_target["cpu_idx"]) + "/online", "w")
            f.write("1\n")
            f.close()
        if not online_flag and not cpu_info_dict[cpu_target["cpu_idx"]]["cpu_online"]:
            continue
        if len(cpu_info_dict[cpu_target["cpu_idx"]]["available_freq"]) == 0:
            cpu_info_dict[cpu_target["cpu_idx"]] = get_cpu_info([cpu_target["cpu_idx"]])[0]
        if min_flag and cpu_target["min_freq"] not in cpu_info_dict[cpu_target["cpu_idx"]]["available_freq"]:
            error_code *= 10
            error_code += 3
            continue
        if max_flag and cpu_target["max_freq"] not in cpu_info_dict[cpu_target["cpu_idx"]]["available_freq"]:
            error_code *= 10
            error_code += 4
            continue
        if min_flag and not max_flag and cpu_target["min_freq"] > cpu_info_dict[cpu_target["cpu_idx"]]["max_freq"]:
            error_code *= 10
            error_code += 5
            continue
        if max_flag and not min_flag and cpu_target["max_freq"] < cpu_info_dict[cpu_target["cpu_idx"]]["min_freq"]:
            error_code *= 10
            error_code += 6
            continue
        if min_flag and max_flag and cpu_target["min_freq"] > cpu_target["max_freq"]:
            error_code *= 10
            error_code += 7
            continue
        if min_flag and not max_flag:
            f = open(cpu_path.format(cpu_target["cpu_idx"]) + "/cpufreq/scaling_min_freq", "w")
            f.write(str(cpu_target["min_freq"]))
            f.close()
        if max_flag and not min_flag:
            f = open(cpu_path.format(cpu_target["cpu_idx"]) + "/cpufreq/scaling_max_freq", "w")
            f.write(str(cpu_target["max_freq"]))
            f.close()
        if min_flag and max_flag:
            if cpu_target["min_freq"] > cpu_info_dict[cpu_target["cpu_idx"]]["max_freq"]:
                f_max = open(cpu_path.format(cpu_target["cpu_idx"]) + "/cpufreq/scaling_max_freq", "w")
                f_max.write(str(cpu_target["max_freq"]))
                f_max.close()
                f_min = open(cpu_path.format(cpu_target["cpu_idx"]) + "/cpufreq/scaling_min_freq", "w")
                f_min.write(str(cpu_target["min_freq"]))
                f_min.close()
            else:
                f_min = open(cpu_path.format(cpu_target["cpu_idx"]) + "/cpufreq/scaling_min_freq", "w")
                f_min.write(str(cpu_target["min_freq"]))
                f_min.close()
                f_max = open(cpu_path.format(cpu_target["cpu_idx"]) + "/cpufreq/scaling_max_freq", "w")
                f_max.write(str(cpu_target["max_freq"]))
                f_max.close()
        if gov_flag:
            if cpu_target["governor"] not in cpu_info_dict[cpu_target["cpu_idx"]]["available_gov"]:
                error_code *= 10
                error_code += 8
                continue
            f = open(cpu_path.format(cpu_target["cpu_idx"]) + "/cpufreq/scaling_governor", "w")
            f.write(cpu_target["governor"])
            f.close()
        error_code *= 10
    if error_code == 0:
        return 0
    return -error_code


def get_gpu_info(min_freq=True, max_freq=True, cur_freq=True, available_freq=True, cur_gov=True,
                 available_gov=True) -> Dict:
    """
    With no arguments, this function reads a series of virtual files
    and return all possible info. If it receives arguments, it will only
    return corresponding values.

    :param bool min_freq: Return minimum frequency or not.
    :param bool max_freq: Return maximum frequency or not.
    :param bool cur_freq: Return current frequency or not.
    :param bool available_freq: Return available frequencies or not.
    :param bool cur_gov: Return current governor or not.
    :param bool available_gov: Return available governors or not.
    :return: A dictionary where gpu information is stored.
    """
    gpu_path = "/sys/devices/gpu.0/devfreq/17000000.gp10b"
    gpu_info = {}
    if min_freq:
        f = open(gpu_path + "/min_freq", "r")
        content = f.read()
        f.close()
        gpu_info["min_freq"] = int(content.rstrip())
    if max_freq:
        f = open(gpu_path + "/max_freq", "r")
        content = f.read()
        f.close()
        gpu_info["max_freq"] = int(content.rstrip())
    if cur_freq:
        f = open(gpu_path + "/cur_freq", "r")
        content = f.read()
        f.close()
        gpu_info["cur_freq"] = int(content.rstrip())
    if available_freq:
        f = open(gpu_path + "/available_frequencies", "r")
        content = f.read()
        f.close()
        gpu_info["available_freq"] = [int(val) for val in content.rstrip().split()]
    if cur_gov:
        f = open(gpu_path + "/governor", "r")
        content = f.read()
        f.close()
        gpu_info["cur_gov"] = content.rstrip()
    if available_gov:
        f = open(gpu_path + "/available_governors", "r")
        content = f.read()
        f.close()
        gpu_info["available_gov"] = content.rstrip().split()
    return gpu_info


def set_gpu_state(gpu_target: Dict) -> int:
    """
    **Note: Governor cannot be set currently. There may be some errors, and the code is commented.**

    This function takes in a dict and sets the gpu parameters accordingly.
    The possible three options are "min_freq", "max_freq", and "governor".

    There are many possible errors and here's a list error code:

    - -1: The specified minimum frequency is not in available frequency list.
    - -2: The specified maximum frequency is not in available frequency list.
    - -3: Only minimum frequency is specified, and it is greater than current maximum frequency.
    - -4: Only maximum frequency is specified, and it is smaller than current minimum frequency.
    - -5: The specified maximum frequency is smaller than the specified minimum frequency.
    - -6: The specified governor is not in available governor list.

    :param Dict gpu_target: A dict which contains gpu parameters.
    :return: Refer to detailed doc.
    """
    gpu_path = "/sys/devices/gpu.0/devfreq/17000000.gp10b"

    gpu_info = get_gpu_info()
    min_flag = "min_freq" in gpu_target
    max_flag = "max_freq" in gpu_target
    gov_flag = "governor" in gpu_target
    if min_flag and gpu_target["min_freq"] not in gpu_info["available_freq"]:
        return -1
    if max_flag and gpu_target["max_freq"] not in gpu_info["available_freq"]:
        return -2
    if min_flag and not max_flag and gpu_target["min_freq"] > gpu_info["max_freq"]:
        return -3
    if max_flag and not min_flag and gpu_target["max_freq"] < gpu_info["min_freq"]:
        return -4
    if min_flag and max_flag and gpu_target["min_freq"] > gpu_target["max_freq"]:
        return -5
    if min_flag and not max_flag:
        f = open(gpu_path + "/min_freq", "w")
        f.write(str(gpu_target["min_freq"]))
        f.close()
    if max_flag and not min_flag:
        f = open(gpu_path + "/max_freq", "w")
        f.write(str(gpu_target["max_freq"]))
        f.close()
    if min_flag and max_flag:
        if gpu_target["min_freq"] > gpu_info["max_freq"]:
            f_max = open(gpu_path + "/max_freq", "w")
            f_max.write(str(gpu_target["max_freq"]))
            f_max.close()
            f_min = open(gpu_path + "/min_freq", "w")
            f_min.write(str(gpu_target["min_freq"]))
            f_min.close()
        else:
            f_min = open(gpu_path + "/min_freq", "w")
            f_min.write(str(gpu_target["min_freq"]))
            f_min.close()
            f_max = open(gpu_path + "/max_freq", "w")
            f_max.write(str(gpu_target["max_freq"]))
            f_max.close()
    if gov_flag:
        if gpu_target["governor"] not in gpu_info["available_gov"]:
            return -6
        # f = open(gpu_path + "/governor", "w")
        # f.write(gpu_target["governor"])
        # f.close()
    return 0
