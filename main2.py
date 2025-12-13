import json
import math
from colorama import Fore, init
import matplotlib.pyplot as plt


DATA_FILE = "data.json"


def load_data():
    with open(DATA_FILE, "r", encoding="utf-8-sig") as data_file:
        data = json.load(data_file)
    return data


def compute_params(data):
    rho_air = data["rho_air"]
    C = data["C"]
    rho_shell = data["rho_shell"]
    d = data["diameter"]

    S = math.pi * d ** 2 / 4
    V = math.pi * d ** 3 / 6
    m = rho_shell * V
    k = C * rho_air * S / (2 * m)

    return k, S, m


def initial_conditions(data):
    t0 = data["t_begin"]
    v0 = data["v0"]
    alpha_rad = math.radians(data["alpha_deg"])

    x1_0 = 0.0
    x2_0 = 0.0
    v1_0 = v0 * math.cos(alpha_rad)
    v2_0 = v0 * math.sin(alpha_rad)

    return t0, x1_0, x2_0, v1_0, v2_0


def runge_kutta_system(data, k):

    g = data["g"]
    h = data["h"]
    t_end = data["t_end"]

    t, x1, x2, v1, v2 = initial_conditions(data)

    n_steps = int((t_end - t) / h)

    result = [(t, x1, x2, v1, v2)]

    for _ in range(n_steps):
        v_abs = math.sqrt(v1 ** 2 + v2 ** 2)

        K1 = h * v1
        L1 = h * v2
        S1 = h * (-k * v_abs * v1)
        Q1 = h * (-k * v_abs * v2 - g)

        v1_2 = v1 + S1 / 2
        v2_2 = v2 + Q1 / 2
        v_abs2 = math.sqrt(v1_2 ** 2 + v2_2 ** 2)

        K2 = h * (v1 + S1 / 2)
        L2 = h * (v2 + Q1 / 2)
        S2 = h * (-k * v_abs2 * v1_2)
        Q2 = h * (-k * v_abs2 * v2_2 - g)

        v1_3 = v1 + S2 / 2
        v2_3 = v2 + Q2 / 2
        v_abs3 = math.sqrt(v1_3 ** 2 + v2_3 ** 2)

        K3 = h * (v1 + S2 / 2)
        L3 = h * (v2 + Q2 / 2)
        S3 = h * (-k * v_abs3 * v1_3)
        Q3 = h * (-k * v_abs3 * v2_3 - g)

        v1_4 = v1 + S3
        v2_4 = v2 + Q3
        v_abs4 = math.sqrt(v1_4 ** 2 + v2_4 ** 2)

        K4 = h * (v1 + S3)
        L4 = h * (v2 + Q3)
        S4 = h * (-k * v_abs4 * v1_4)
        Q4 = h * (-k * v_abs4 * v2_4 - g)

        x1 = x1 + (K1 + 2 * K2 + 2 * K3 + K4) / 6
        x2 = x2 + (L1 + 2 * L2 + 2 * L3 + L4) / 6
        v1 = v1 + (S1 + 2 * S2 + 2 * S3 + S4) / 6
        v2 = v2 + (Q1 + 2 * Q2 + 2 * Q3 + Q4) / 6
        t = t + h

        result.append((t, x1, x2, v1, v2))

    return result


def extract_series(result):
    t_list = [p[0] for p in result]
    x1_list = [p[1] for p in result]
    x2_list = [p[2] for p in result]
    v1_list = [p[3] for p in result]
    v2_list = [p[4] for p in result]
    return t_list, x1_list, x2_list, v1_list, v2_list


def plot_two_series(x1, y1, x2, y2, fig_name, title, x_label, y_label, label1, label2):
    plt.figure(fig_name)
    plt.plot(x1, y1, label=label1)
    plt.plot(x2, y2, label=label2)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.title(title)
    plt.grid(True)
    plt.legend()


def truncate_at_ground(result):
    truncated = [result[0]]
    for i in range(1, len(result)):
        prev = result[i - 1]
        curr = result[i]
        if prev[2] >= 0 and curr[2] <= 0:
            denominator = prev[2] - curr[2]
            factor = 0 if denominator == 0 else prev[2] / denominator
            t_ground = prev[0] + (curr[0] - prev[0]) * factor
            x1_ground = prev[1] + (curr[1] - prev[1]) * factor
            v1_ground = prev[3] + (curr[3] - prev[3]) * factor
            v2_ground = prev[4] + (curr[4] - prev[4]) * factor
            truncated.append((t_ground, x1_ground, 0.0, v1_ground, v2_ground))
            return truncated
        truncated.append(curr)
    return truncated


def compute_flight_metrics(result):
    max_height = max(point[2] for point in result)
    flight_time = result[-1][0]
    distance = result[-1][1]
    return flight_time, distance, max_height


def main():
    data = load_data()

    print(f"{Fore.LIGHTBLUE_EX}Исходные данные\n")
    print(f"Интервал времени: [{data['t_begin']}, {data['t_end']}]")
    print(f"Шаг по времени h: {data['h']}")
    print(f"Начальная скорость v0: {data['v0']} м/с")
    print(f"Угол наклона ствола α: {data['alpha_deg']}°")
    print(f"Диаметр снаряда d: {data['diameter']} м\n")

    k_phys, S, m = compute_params(data)

    print(f"{Fore.GREEN}Рассчитанные параметры:")
    print(f"{Fore.WHITE}Площадь поперечного сечения S: {S:.4e} м²")
    print(f"Масса снаряда m: {m:.4f} кг")
    print(f"Коэффициент сопротивления k (по формуле CρS/2m): {Fore.CYAN}{k_phys:.4e}{Fore.RESET}\n")

    print(f"{Fore.GREEN}Решение методом Рунге–Кутты 4 порядка при k = 0")
    result_k0_raw = runge_kutta_system(data, k=0.0)
    result_k0 = truncate_at_ground(result_k0_raw)
    t0, x1_0, x2_0, v1_0, v2_0 = extract_series(result_k0)
    print(f"{Fore.WHITE}Конечная точка (t={t0[-1]}): x1 = {x1_0[-1]:.2f} м, x2 = {x2_0[-1]:.2f} м")

    print(f"\n{Fore.GREEN}Решение методом Рунге–Кутты 4 порядка при k = CρS / (2m)")
    result_k_raw = runge_kutta_system(data, k=k_phys)
    result_k = truncate_at_ground(result_k_raw)
    t1, x1_1, x2_1, v1_1, v2_1 = extract_series(result_k)
    print(f"{Fore.WHITE}Конечная точка (t={t1[-1]}): x1 = {x1_1[-1]:.2f} м, x2 = {x2_1[-1]:.2f} м")

    flight_time_k0, distance_k0, max_height_k0 = compute_flight_metrics(result_k0)
    flight_time_k, distance_k, max_height_k = compute_flight_metrics(result_k)

    print(f"{Fore.GREEN}\nХарактеристики полета при k = 0")
    print(f"{Fore.WHITE}Общее время полета: {flight_time_k0:.2f} с")
    print(f"Общая дальность полета: {distance_k0:.2f} м")
    print(f"Максимальная высота: {max_height_k0:.2f} м")

    print(f"{Fore.GREEN}\nХарактеристики полета при k = CρS / (2m)")
    print(f"{Fore.WHITE}Общее время полета: {flight_time_k:.2f} с")
    print(f"Общая дальность полета: {distance_k:.2f} м")
    print(f"Максимальная высота: {max_height_k:.2f} м")

    print(f"\n{Fore.GREEN}Построение графиков траекторий\n")

    plot_two_series(t0, x1_0, t1, x1_1, fig_name="x1(t)", title="График x1(t)", x_label="t, с", y_label="x1(t), м", label1="x1(t), k = 0", label2="x1(t), k = CρS/2m")
    plot_two_series(t0, x2_0, t1, x2_1, fig_name="x2(t)", title="График x2(t)", x_label="t, с", y_label="x2(t), м", label1="x2(t), k = 0", label2="x2(t), k = CρS/2m")
    plot_two_series(x1_0, x2_0, x1_1, x2_1, fig_name="x2(x1)", title="Траектория полета x2(x1)", x_label="x1, м", y_label="x2, м", label1="траектория, k = 0", label2="траектория, k = CρS/2m")
    plt.show()


if __name__ == "__main__":
    init(autoreset=True)
    main()
