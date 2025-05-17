import serial
import time
import pandas as pd
from datetime import datetime

PORT = "COM5"
BAUDRATE = 57600
DURATION_SECONDS = 3600  # 1小时
OUTPUT_FILE = "eeg_brainwaves_1hour.csv"

def read_3byte(data, idx):
    return (data[idx] << 16) + (data[idx+1] << 8) + data[idx+2]

# 从串口实时读取脑波数据包，解析关键频段
def collect_full_brainwave_packets():
    #  初始化串口与缓冲区
    ser = serial.Serial(PORT, BAUDRATE, timeout=0.1)
    buffer = bytearray()
    SYNC = b'\xAA\xAA'  # 数据包起始标志
    all_data = []

    print("开始采集完整脑波数据包，持续 1 小时... 按 Ctrl+C 可中断")
    start_time = time.time()

    try:
        while time.time() - start_time < DURATION_SECONDS:
            # 数据读取与拼接
            byte = ser.read()  # 每次读取1字节
            if not byte:
                continue
            buffer += byte

            while len(buffer) >= 36:
                #  数据包识别与校验
                if buffer[0:2] == SYNC:
                    payload_len = buffer[2]
                    total_len = 3 + payload_len + 1  # 含头部、负载、校验
                    if len(buffer) < total_len:
                        break

                    payload = buffer[3:3+payload_len]
                    checksum = buffer[3+payload_len]
                    if (0xFF - (sum(payload) & 0xFF)) & 0xFF == checksum:
                        # 检查是否包含完整脑波字段（>=36 字节）
                        if payload_len >= 32:
                            try:
                                # 脑波频段提取
                                record = {
                                    'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f'),
                                    'delta': read_3byte(payload, 7),
                                    'theta': read_3byte(payload, 10),
                                    'low_alpha': read_3byte(payload, 13),
                                    'high_alpha': read_3byte(payload, 16),
                                    'low_beta': read_3byte(payload, 19),
                                    'high_beta': read_3byte(payload, 22),
                                    'low_gamma': read_3byte(payload, 25),
                                    'mid_gamma': read_3byte(payload, 28),
                                }
                                all_data.append(record)
                                print(f"{record['timestamp']} [delta={record['delta']}]")
                            except:
                                pass
                    buffer = buffer[total_len:]
                else:
                    buffer = buffer[1:]

    except KeyboardInterrupt:
        print("用户中断采集。")

    finally:
        ser.close()
        df = pd.DataFrame(all_data)
        df.to_csv(OUTPUT_FILE, index=False)
        print(f"采集完成，共记录 {len(df)} 条完整脑波包，保存至 {OUTPUT_FILE}")

if __name__ == '__main__':
    collect_full_brainwave_packets()
