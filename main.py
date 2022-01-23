import re
import sys

env = {
    "print": print,
    "input": input,
}

types = ["int", "flt", "str", "bin"]

RESERVED = 'RESERVED'
TYPE     = "TYPE"
INT      = 'INT'
FLT      = 'FLT'
STR      = 'STR'
BIN      = 'BIN'
VOID     = 'VOID'
ID       = 'ID'
EOF      = 'EOF'
OP       = 'OP'

token_exprs = [
    (r'(["])((?:(?=(?:\\)*)\\.|.)*?)\1', STR),
    (r'[ \n\t]+',              None),
    (r'#[^\n]*',               None),
    (r'==', OP),
    (r'\=', RESERVED),
    (r'\(', RESERVED),
    (r'\)', RESERVED),
    (r';', RESERVED),
    (r'\+', OP),
    (r'-', OP),
    (r'\*', OP),
    (r'/', OP),
    (r'\%', OP),
    (r'<=', OP),
    (r'<', OP),
    (r'>=', OP),
    (r'>', OP),
    (r'!=', OP),
    (r'{', RESERVED),
    (r'}', RESERVED),
    (r'void', TYPE),
    (r'int', TYPE),
    (r'flt', TYPE),
    (r'str', TYPE),
    (r'bin', TYPE),
    (r'fun', RESERVED),
    (r'if', RESERVED),
    (r'return', RESERVED),
    (r'else', RESERVED),
    (r'while', RESERVED),
    (r'true|false', BIN),
    (r'use', RESERVED),
    (r'[+-]?([0-9]*[.])?[0-9]+',FLT),
    (r'[0-9]+',                INT),
    (r'[A-Za-z][A-Za-z0-9_]*', ID),
]
def lex(characters):
    pos = 0
    line = 1
    tokens = []
    while pos < len(characters):
        match = None
        for token_expr in token_exprs:
            pattern, tag = token_expr
            regex = re.compile(pattern)
            match = regex.match(characters, pos)
            if match:
                text = match.group(0)
                if text == "\n":
                    line += 1
                elif tag:
                    token = (text, tag, pos, line)
                    tokens.append(token)
                break
        if not match:
            sys.stderr.write('Illegal character: %s\\n' % characters[pos])
            sys.exit(1)
        else:
            pos = match.end(0)
    tokens.append((None, EOF, pos, line))
    return tokens

class Number:
    def __init__(self, value):
        self.value = value

    def resolve(self):
        return self.value

class Float:
    def __init__(self, value):
        self.value = value

    def resolve(self):
        return self.value

class String:
    def __init__(self, value):
        self.value = value

    def resolve(self):
        return self.value

class Bool:
    def __init__(self, value):
        self.value = value

    def resolve(self):
        return self.value

class Void:
    def __init__(self, value):
        self.value = value

    def resolve(self):
        return self.value

class VariableLookup:
  def __init__(self, name):
    self.name = name
    
  def resolve(self):
    return env[self.name]

class BinOp:
    def __init__(self, left, op, right):
        self.left = left
        self.op = op
        self.right = right

    def resolve(self):
        match self.op:
            case '+':
                return self.left.resolve() + self.right.resolve()
            case '-':
                return self.left.resolve() - self.right.resolve()
            case '*':
                return self.left.resolve() * self.right.resolve()
            case '/':
                return self.left.resolve() / self.right.resolve()
            case '%':
                return self.left.resolve() % self.right.resolve()
            case '==':
                return self.left.resolve() == self.right.resolve()
            case '!=':
                return self.left.resolve() != self.right.resolve()
            case '<=':
                return self.left.resolve() <= self.right.resolve()
            case '>=':
                return self.left.resolve() >= self.right.resolve()
            case '<':
                return self.left.resolve() < self.right.resolve()
            case '>':
                return self.left.resolve() > self.right.resolve()

class VarDecl:
    def __init__(self, name, value):
        self.name = name
        self.value = value

    def resolve(self):
        env[self.name] = self.value.resolve()

class FuncCall:
    def __init__(self, name, operands):
        self.name = name
        self.operands = operands

    def resolve(self):
        return env[self.name](*[i.resolve() for i in self.operands])

class IfStmt:
    def __init__(self, cond, body, elseb):
        self.cond = cond
        self.body = body
        self.elseb = elseb

    def resolve(self):
        if self.cond.resolve():
            self.body.resolve()
        elif self.elseb is not None:
            self.elseb.resolve()

class WhileStmt:
    def __init__(self, cond, body):
        self.cond = cond
        self.body = body

    def resolve(self):
        while self.cond.resolve():
            self.body.resolve()

class ReturnStmt:
    def __init__(self, value):
        self.value = value

    def resolve(self):
        return self.value.resolve()

class Block:
    def __init__(self, body):
        self.body = body

    def resolve(self):
        for i in self.body:
            if isinstance(i, ReturnStmt):
                return i.resolve()
            i.resolve()

class FuncBody:
    def __init__(self, name, operands, body):
        self.name = name
        self.operands = operands
        self.body = body

    def resolve(self):
        o = self.operands
        b = self.body
        def _t(*args):
            # print(o)
            for i in range(len(args)):
                env[o[i][1]] = args[i]
            return b.resolve()
        env[self.name] = _t

class Parser():
    def __init__(self, token):
        self.tokens = token

    
    def peek(self, ttype):
        if ttype not in (INT, FLT, STR, BIN, ID, EOF, OP, VOID, TYPE):
            return self.tokens[0][0] == ttype
            
        else:
            return self.tokens[0][1] == ttype

    def eat(self, ttype):
        assert self.peek(ttype), ttype
        return self.tokens.pop(0)[0]

    def number(self):
        if self.peek("INT"):
            return Number(int(self.eat("INT")))
        elif self.peek("FLT"):
            return Float(float(self.eat("FLT")))

    def string(self):
        if self.peek("STR"):
            return String(self.eat("STR").strip('"'))

    def tbool(self):
        if self.peek("BIN"):
            return Bool(self.eat("BIN"))

    def void(self):
        if self.peek("VOID"):
            return Void(self.eat("VOID"))
 
    def funccall(self):
        function_name = self.eat("ID")
        args = []
        self.eat("(")
        while not self.peek(')'):
            args.append(self.expr())
        self.eat(')')
        return FuncCall(function_name, args)

    def funcbody(self):
        self.eat("fun")
        function_type = self.eat("TYPE")
        function_name = self.eat("ID")
        self.eat("(")
        operands = []
        while not self.peek(")"):
            operands.append([self.eat("TYPE"), self.eat("ID")])

        self.eat(")")
        self.eat("{")
        body = []
        while not self.peek("}"):
            body.append(self.stmt())
        self.eat("}")
        return FuncBody(function_name, operands, Block(body))

    def stmt(self):
        result = None
        if self.peek("return"):
            self.eat("return")
            result =  ReturnStmt(self.expr())
        elif self.peek("TYPE"):
            result = self.varDecl()
        elif self.peek("fun"):
            result = self.funcbody()
        elif self.peek("if"):
            result = self.ifs()
        elif self.peek("while"):
            result = self.whiles()
        else:
            result = self.funccall()
        self.eat(";")
        return result

    def ifs(self):
        self.eat('if')
        self.eat('(')
        cond = self.expr()
        self.eat(')')
        self.eat('{')
        body = []
        while not self.peek('}'):
            body.append(self.stmt())
        self.eat('}')
        if self.peek('else'):
            self.eat('else')
            self.eat('{')
            ebody = []
            while not self.peek('}'):
                ebody.append(self.stmt())
            self.eat('}')
            return IfStmt(cond, Block(body), Block(ebody))
        return IfStmt(cond, Block(body), None)

    def whiles(self):
        self.eat('while')
        self.eat('(')
        cond = self.expr()
        self.eat(')')
        self.eat('{')
        body = []
        while not self.peek('}'):
            body.append(self.stmt())
        self.eat('}')
        if self.peek('else'):
            self.eat('else')
            self.eat('{')
            ebody = []
            while not self.peek('}'):
                ebody.append(self.stmt())
            self.eat('}')
            return WhileStmt(cond, Block(body))
        return WhileStmt(cond, Block(body))   

    def root(self):
        body = []
        while not self.peek("EOF"):
            body.append(self.stmt())

        return body

    def binop(self):
        left = self.expr()
        op = self.eat("OP")
        right = self.expr()
        return BinOp(left, op, right)

    def expr(self):
        if self.peek("ID"):
            if self.tokens[1][0] == "(":
                return self.funccall()
            elif self.tokens[1][1] != "OP":
                return VariableLookup(self.eat("ID"))
            else:
                left = VariableLookup(self.eat("ID"))
                op = self.eat("OP")
                right = self.expr()
                return BinOp(left, op, right)
            
        elif self.peek("INT") or self.peek("FLT"):
            if self.tokens[1][1] != "OP":
                return self.number()
            else:
                left = self.number()
                op = self.eat("OP")
                right = self.expr()
                return BinOp(left, op, right)
        elif self.peek("STR"):
            return self.string()
        elif self.peek("BIN"):
            return self.tbool()
        else:
            return self.binop()
    
    def varDecl(self):
        ttype = self.eat("TYPE")
        name = self.eat("ID")
        self.eat("=")
        value = self.expr()
        return VarDecl(name, value)

# open file if arg is given
if len(sys.argv) > 1:
    file = open(sys.argv[1], 'r')
    characters = file.read()
    parser = Parser(lex(characters))
    for i in parser.root():
        i.resolve()
else:
    while True:
        try:
            inp = input("pocketscript> ").strip()
            parser = Parser(lex(inp))
            for i in parser.root():  
                result = i.resolve()
                if result != None:
                    print(result)
        except Exception as e:
            print("Syntax Error -> Missing " + str(e) + " on line " + str(parser.tokens[0][3]))
        