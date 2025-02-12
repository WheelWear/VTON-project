# def count_lines(file_path):
#     with open(file_path, "r") as f:
#         return sum(1 for _ in f)

# if __name__ == "__main__":
#     file_path = "train_pairs.txt"
#     line_count = count_lines(file_path)
#     print(f"Number of lines in {file_path}: {line_count}")

# def count_different_pairs(file_path):
#     diff_count = 0
#     with open(file_path, "r") as f:
#         for line in f:
#             tokens = line.strip().split()
#             if len(tokens) == 2 and tokens[0] != tokens[1]:
#                 diff_count += 1
#     return diff_count

# if __name__ == "__main__":
#     file_path = "train_pairs.txt"
#     diff_pairs_count = count_different_pairs(file_path)
#     print(f"Number of pairs with different strings in {file_path}: {diff_pairs_count}")

def count_left_in_all_right(file_path):
    """
    파일의 각 라인에서 왼쪽 토큰이 전체 오른쪽 토큰 요소 집합에 존재하는 경우를 카운트합니다.
    """
    left_tokens = []
    right_tokens = set()
    
    with open(file_path, "r") as f:
        for line in f:
            tokens = line.strip().split()
            if len(tokens) == 2:
                left, right = tokens
                left_tokens.append(left)
                right_tokens.add(right)
    
    count = 0
    for left in left_tokens:
        if left in right_tokens:
            count += 1
    return count, len(left_tokens)

if __name__ == "__main__":
    file_path = "train_pairs.txt"
    count, total = count_left_in_all_right(file_path)
    print(f"왼쪽 토큰이 오른쪽 토큰 집합 내에 있는 경우: {count} (총 {total}개 중)")