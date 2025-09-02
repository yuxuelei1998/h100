def extract_cmpeq_results(input_file, output_file=None):
    """
    提取CMPEQ指令中比较Operand A和Operand C为0x00000000和0x80000000的结果行
    
    参数:
        input_file (str): 输入文件路径
        output_file (str, optional): 输出文件路径，若为None则仅打印结果
    """
    # 目标比较值（不区分顺序）
    target_values = {'0x00000000', '0x80000000'}
    matched_lines = []
    
    try:
        with open(input_file, 'r') as infile:
            # 读取表头并解析列索引
            header = infile.readline().strip()
            if not header:
                print("错误：输入文件为空")
                return
            
            columns = [col.strip() for col in header.split(',')]
            print(f"检测到表头列：{columns}")
            
            # 确定关键列的索引（修改为A和C比较）
            try:
                opcode_idx = columns.index('Opcode')
                op_a_idx = columns.index('Operand A')  # 比较对象A
                op_c_idx = columns.index('Operand C')  # 比较对象C（原脚本为Operand B，此处修改）
                result_idx = columns.index('Result')
            except ValueError as e:
                print(f"错误：表头中未找到目标列 - {e}")
                return
            
            # 遍历数据行筛选符合条件的记录
            line_count = 0
            for line in infile:
                line = line.strip()
                if not line:
                    continue
                line_count += 1
                data = [item.strip() for item in line.split(',')]
                
                # 检查列索引是否有效
                if len(data) <= max(opcode_idx, op_a_idx, op_c_idx, result_idx):
                    print(f"警告：第{line_count}行列数不足，跳过")
                    continue
                
                # 筛选条件：Opcode为CMPEQ，且A和C为目标值（不限制顺序）
                opcode = data[opcode_idx]
                op_a = data[op_a_idx]
                op_c = data[op_c_idx]
                
                if opcode == 'CMPEQ' and {op_a, op_c} == target_values:
                    matched_lines.append(line)
                    print(f"找到匹配行 {line_count}: {line}")
            
            # 输出结果处理
            if not matched_lines:
                print("未找到符合条件的CMPEQ指令行")
                return
            
            print(f"\n共找到 {len(matched_lines)} 行符合条件的指令")
            
            # 保存到文件（如果指定输出路径）
            if output_file:
                with open(output_file, 'w') as outfile:
                    outfile.write(header + '\n')  # 写入表头
                    for line in matched_lines:
                        outfile.write(line + '\n')
                print(f"结果已保存至：{output_file}")
    
    except FileNotFoundError:
        print(f"错误：文件 {input_file} 未找到")
    except Exception as e:
        print(f"处理过程中发生错误：{e}")


if __name__ == "__main__":
    # 输入文件路径（根据实际文件修改）
    input_file = "fp32_boundary_test_cases_output.txt"
    # 输出文件路径（可选）
    output_file = "cmpeq_results_FP32.txt"
    
    # 调用函数执行提取
    extract_cmpeq_results(input_file, output_file)