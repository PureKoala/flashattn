import re
import openpyxl

# 从文件中读取文本内容
def read_text_from_file(file_path):
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            return file.read()
    except FileNotFoundError:
        print(f"文件 {file_path} 未找到。")
        return None

# 提取数据
def extract_data(text):
    all_results = {}
    # 匹配 Begin to prefill 部分
    prefill_matches = re.findall(r'================ Begin to prefill N = (\d+), d = (\d+) ================(.*?)(?================= Begin to prefill N = \d+, d = \d+ ================|$)', text, re.DOTALL)
    for N, d, prefill_match in prefill_matches:
        key = (int(N), int(d))
        all_results[key] = {'prefill': [], 'decoding_rounds': {}}

        # 匹配 Prefill layer 部分
        prefill_layer_matches = re.findall(r'======== Prefill layer (\d+) ========(.*?)(?========= (Prefill layer|Decoding round|Decoding layer|Begin to decoding))', prefill_match, re.DOTALL)
        for layer_num, layer_content, _ in prefill_layer_matches:
            times = re.findall(r'\bTime\b.*?(\d+\.\d+)', layer_content)
            times = [float(time) for time in times]
            result = ['', 'Prefill layer', int(layer_num), ''] + times
            all_results[key]['prefill'].append(result)
        
        # 匹配 Decoding 部分
        decoding_match = re.search(r'================ Begin to decoding ================(.*)', prefill_match, re.DOTALL)
        if decoding_match:
            decoding_content = decoding_match.group(1)

            # 匹配 Decoding round 部分
            decoding_round_matches = re.findall(r'======== Decoding round (\d+) ========(.*?)(?========= (Prefill layer|Decoding round|$))', decoding_content, re.DOTALL)
            for round_num, round_content, _ in decoding_round_matches:
                round_key = int(round_num)
                all_results[key]['decoding_rounds'][round_key] = []

                time = re.search(r'\bTime\b.*?(\d+\.\d+)', round_content)
                if time:
                    result = ['Decoding', '', round_key, float(time.group(1))]
                    all_results[key]['decoding_rounds'][round_key].append(result)

                # 匹配 Decoding layer 部分
                decoding_layer_matches = re.findall(r'======== Decoding layer (\d+) ========(.*?)(?========= (Prefill layer|Decoding round|Decoding layer|$))', round_content, re.DOTALL)
                for layer_num, layer_content, _ in decoding_layer_matches:
                    times = re.findall(r'\bTime\b.*?(\d+\.\d+)', layer_content)
                    times = [float(time) for time in times]
                    result = ['Decoding layer', int(layer_num), ''] + times
                    all_results[key]['decoding_rounds'][round_key].append(result)

    return all_results

# 将提取的数据写入 Excel 文件
def write_to_excel(results, output_file):
    workbook = openpyxl.Workbook()
    sheet = workbook.active
    sheet.title = 'Results'
    # 添加表头
    headers = ['N', 'd', 'Outer', 'Inner', 'Count'] + \
                [
                    'word embedding',
                    'attention pre layer norm',
                    'Q, K, V',
                    'S',
                    'P',
                    'O',
                    'attention block',
                    'attention residual connection',
                    'MLP pre layer norm',
                    'H1',
                    'Gelu',
                    'H2',
                    'MLP residual connection'
                ]
    sheet.append(headers)
    # 添加数据
    for (N, d), data in results.items():
        sheet.append([N, d, 'Prefill'])
        for row in data['prefill']:
            sheet.append(['', ''] + row)
        sheet.append([])
        for round_key, round_data in data['decoding_rounds'].items():
            i = 0
            for row in round_data:
                if i == 0:
                    sheet.append(['', ''] + row + ['' for _ in range(13 - len(row))])
                    i += 1
                else:
                    sheet.append(['', '', ''] + row + ['' for _ in range(13 - len(row))])
        sheet.append([])
    # 调整列宽
    for col in sheet.iter_cols():
        max_length = 0
        col_letter = col[0].column_letter
        for cell in col:
            try:
                max_length = max(max_length, len(str(cell.value)))
            except:
                pass
        adjusted_width = (max_length + 2) * 1.2
        sheet.column_dimensions[col_letter].width = adjusted_width
    # 保存文件
    workbook.save(output_file)

if __name__ == "__main__":
    input_file = 'build/static.txt'  # 替换为你的 txt 文件路径
    output_file = 'output.xlsx'  # 替换为你想要保存的 xlsx 文件路径
    text = read_text_from_file(input_file)
    if text:
        results = extract_data(text)
        with open('debug_output.txt', 'w', encoding='utf-8') as debug_file:
            for key, value in results.items():
                debug_file.write(f"{key}: {value}\n")
        write_to_excel(results, output_file)
        print(f"数据已成功写入 {output_file}。")