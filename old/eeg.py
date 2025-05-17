import csv
import pandas as pd
# 文件路径
input_file = "data/2.txt"
output_file = "data/eeg_data_updated.csv"

# 重新解析数据，只保留有用字段
def parse_thinkgear_data_filtered(input_file, output_file):
    # 读取输入文件
    with open(input_file, "r") as file:
        hex_data = file.read().replace(" ", "").replace("\n", "")

    # 解析数据
    records = []
    sync_bytes = "AAAA"
    index = 0
    data_len = len(hex_data)

    while index < data_len - 4:
        # 找到同步字节 0xAA 0xAA
        if hex_data[index:index + 4] == sync_bytes:
            try:
                # 读取 payload 长度
                plength = int(hex_data[index + 4:index + 6], 16)
                payload_start = index + 6
                payload_end = payload_start + plength * 2

                # 检查数据是否完整
                if payload_end + 2 > data_len:
                    break

                # 读取 payload 和校验和
                payload = hex_data[payload_start:payload_end]
                checksum = int(hex_data[payload_end:payload_end + 2], 16)

                # 校验和验证
                payload_bytes = bytes.fromhex(payload)
                calculated_checksum = (~sum(payload_bytes) & 0xFF)
                if calculated_checksum != checksum:
                    index += 2
                    continue

                # 解析 payload 数据
                payload_index = 0
                record = {
                    "secs": len(records),
                    "attention": 0,
                    "mytheta": 0,
                    "mydelta": 0,
                    "mylow_alpha": 0,
                    "myhigh_alpha": 0,
                    "mylow_beta": 0,
                    "myhigh_beta": 0,
                }

                while payload_index < len(payload):
                    code = int(payload[payload_index:payload_index + 2], 16)
                    payload_index += 2

                    if code == 0x04:  # Attention
                        record["attention"] = int(payload[payload_index:payload_index + 2], 16)
                        payload_index += 2
                    elif code == 0x83:  # EEG power values
                        length = int(payload[payload_index:payload_index + 2], 16)
                        payload_index += 2
                        eeg_data = payload[payload_index:payload_index + length * 2]
                        record["mydelta"] = int(eeg_data[0:6], 16)
                        record["mytheta"] = int(eeg_data[6:12], 16)
                        record["mylow_alpha"] = int(eeg_data[12:18], 16)
                        record["myhigh_alpha"] = int(eeg_data[18:24], 16)
                        record["mylow_beta"] = int(eeg_data[24:30], 16)
                        record["myhigh_beta"] = int(eeg_data[30:36], 16)
                        payload_index += length * 2
                    else:
                        payload_index += 2  # 跳过未知代码

                # 添加记录
                records.append(record)

            except Exception as e:
                index += 2
                continue

        index += 2

    # 写入 CSV 文件
    with open(output_file, mode="w", newline="") as csv_file:
        fieldnames = ["secs", "attention", "mytheta", "mydelta", "mylow_alpha", "myhigh_alpha", "mylow_beta", "myhigh_beta"]
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)

        writer.writeheader()
        writer.writerows(records)

# 生成精简版数据
filtered_output_file = "data/eeg_data_filtered.csv"
parse_thinkgear_data_filtered(input_file, filtered_output_file)

# 读取已解析的 CSV 文件
filtered_data = pd.read_csv("data/eeg_data_filtered.csv")

# 过滤掉指定列 (mytheta, mydelta, mylow_alpha, myhigh_alpha, mylow_beta, myhigh_beta) 全为 0 的行
columns_to_check = ["mytheta", "mydelta", "mylow_alpha", "myhigh_alpha", "mylow_beta", "myhigh_beta"]
valid_data_final = filtered_data.loc[(filtered_data[columns_to_check] != 0).any(axis=1)]
# 重新排列 secs 列，从 1 开始递增
valid_data_final.reset_index(drop=True, inplace=True)
valid_data_final["secs"] = valid_data_final.index + 1

# 保存更新后的数据
final_reindexed_output_file = "data/eeg_data_final_reindexed.csv"
valid_data_final.to_csv(final_reindexed_output_file, index=False)




