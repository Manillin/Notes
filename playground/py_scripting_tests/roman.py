def evalRPN(tokens) -> int:
    operands = ('+', '-', '*', '/')
    stack = []
    for char in tokens:
        if char in operands:
            op1 = stack.pop()
            op2 = stack.pop()
            if char == '+':
                stack.append((op1+op2))
            elif char == '-':
                stack.append((op2-op1))
            elif char == '*':
                stack.append((op1*op2))
            else:
                division = int(op2/op1)
                stack.append(division)
        else:
            stack.append(int(char))
    return stack.pop()


def generateParenthesis(self, n: int):
    stack = []
    res = []

    def backtrack(openN, closedN):
        if openN == closedN == n:
            res.append("".join(stack))
            return

        if openN < n:
            stack.append("(")
            backtrack(openN+1, closedN)
            stack.pop()
        if closedN < openN:
            stack.append(")")
            backtrack(openN, closedN+1)
            stack.pop()
    backtrack(0, 0)
    return res
