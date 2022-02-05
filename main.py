from random import randint, randrange, uniform
import re
import sys
from time import sleep

from colorama import Fore, Style, init

repl = False

def errorStatement(msg):
    print(Fore.RED, "\b" + "SYNTAX ERROR: " + msg + Style.RESET_ALL)
    if not repl:
        exit(1)

def psUserException(msg):
    print(Fore.RED, "\b" + msg + Style.RESET_ALL)
    if not repl:
        exit(1)
        
init()

env = {
    "print": lambda x: print(x, end=""),
    "println": print,
    "sleep": lambda s: sleep(float(s)),
    "input": input,
    "randint": lambda x, y: int(randint(x, y)),
    "randflt": lambda x, y: float(uniform(x, y)),
    "rnd": round,
    "error": psUserException,
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
VAROP    = 'VAROP'

token_exprs = [
    (r'(["])((?:(?=(?:\\)*)\\.|.)*?)\1', STR),
    (r'[ \n\t]+',              None),
    (r'#[^\n]*',               None),
    (r'\+=', VAROP),
    (r'-=', VAROP),
    (r'/=', VAROP),
    (r'\*=', VAROP),
    (r'==', OP),
    (r'\=', VAROP),
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
    (r'[+-]?([0-9]*[.])[0-9]+',FLT),
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

def typeStr(v):
    if isinstance(v, int): return "int"
    if isinstance(v, float): return "flt"
    if isinstance(v, str): return "str"
    if isinstance(v, bool): return "bin"


class Number:
    def __init__(self, value):
        self.value = value

    def resolve(self):
        return int(self.value)

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

class VarOp:
    def __init__(self, left, op, right):
        self.left = left
        self.op = op
        self.right = right

    def resolve(self):
        match self.op:
            case '=':
                v = self.right.resolve()
                match typeStr(v):
                    case "int":
                        assert isinstance(v, int), errorStatement("Type mismatch in variable reference")
                    case "flt":
                        assert isinstance(v, float), errorStatement("Type mismatch in variable reference")
                    case "str":
                        assert isinstance(v, str), errorStatement("Type mismatch in variable reference")
                    case "bin":
                        assert isinstance(v, bool), errorStatement("Type mismatch in variable reference")

                if self.left in env.keys() : env[self.left] = v
            case '+=':
                env[self.left] += self.right.resolve()
            case '-=':
                env[self.left] -= self.right.resolve()
            case '*=':
                env[self.left] *= self.right.resolve()
            case '/=':
                env[self.left] /= self.right.resolve()

class VarDecl:
    def __init__(self, name, value, typee):
        self.name = name
        self.value = value
        self.typee = typee

    def resolve(self):
        v = self.value.resolve()
        match self.typee:
            case "int":
                assert isinstance(v, int), errorStatement("Type mismatch in variable reference")
            case "flt":
                assert isinstance(v, float), errorStatement("Type mismatch in variable reference")
            case "str":
                assert isinstance(v, str), errorStatement("Type mismatch in variable reference")
            case "bin":
                assert isinstance(v, bool), errorStatement("Type mismatch in variable reference")

        env[self.name] = v

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

class Use:
    def __init__(self, fn):
        self.fn = fn

    def resolve(self):
        env = {}
        for i in Parser(lex(open(self.fn.resolve()).read())).root():
            i.resolve()
        globals()["env"].update(env)
        


class FuncBody:
    def __init__(self, name, operands, body, typee):
        self.name = name
        self.operands = operands
        self.body = body
        self.typee = typee

    def resolve(self):
        o = self.operands
        b = self.body
        t = self.typee
        def _t(*args):
            for i in range(len(args)):
                if isinstance(args[i], int): 
                    assert o[i][0] == "int", "Type mismatch"
                elif isinstance(args[i], float):
                    assert o[i][0] == "flt", "Type mismatch"
                elif isinstance(args[i], str):
                    assert o[i][0] == "str", "Type mismatch"
                elif isinstance(args[i], bool):
                    assert o[i][0] == "bin", "Type mismatch"
                else:
                    print(args[i])

                env[o[i][1]] = args[i]
                
            v = b.resolve()
            
            if isinstance(v, int): 
                    assert t == "int", "Type mismatch"
            elif isinstance(v, float):
                assert t == "flt", "Type mismatch"
            elif isinstance(v, str):
                assert t == "str", "Type mismatch"
            elif isinstance(v, bool):
                assert t == "bin", "Type mismatch"

            return v

        env[self.name] = _t

class Parser():
    def __init__(self, token):
        self.tokens = token

    
    def peek(self, ttype):
        if ttype not in (INT, FLT, STR, BIN, ID, EOF, OP, VAROP, VOID, TYPE):
            return self.tokens[0][0] == ttype
            
        else:
            return self.tokens[0][1] == ttype

    def eat(self, ttype):
        # assert self.peek(ttype), ttype
        if not self.peek(ttype):
            errorStatement("Missing %s" %ttype)
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
        return FuncBody(function_name, operands, Block(body), function_type)

    def stmt(self):
        result = None
        if self.peek("ID"):
            if self.tokens[1][0] == "(":
                result = self.funccall()
            
            elif self.tokens[1][1] == VAROP:
                left = self.eat('ID')
                op = self.eat('VAROP')
                right = self.expr()
                print(left,op,right)
                result = VarOp(left, op, right)
        elif self.peek("return"):
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
        elif self.peek("use"):
            result = self.use()
        else:
            errorStatement("PARSER ERROR" + self.tokens)
            # result = self.funccall()
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

    def expr(self, binop=False):
        if not binop and self.tokens[1][1] == OP:
            left = self.expr(True)
            op = self.eat('OP')
            right = self.expr()
            return BinOp(left, op, right)

        if self.peek("ID"):
            if self.tokens[1][0] == "(":
                return self.funccall()
            return VariableLookup(self.eat("ID"))
            
        elif self.peek("INT") or self.peek("FLT"):
            if self.tokens[1][1] != "OP":
                return self.number()
        elif self.peek("STR"):
            return self.string()
        elif self.peek("BIN"):
            return self.tbool()
        else:
            raise Exception("wtf are you doing: CODE ID10T")
        
       
    
    def varDecl(self):
        ttype = self.eat("TYPE")
        name = self.eat("ID")
        self.eat("=")
        value = self.expr()
        return VarDecl(name, value, ttype)

    def use(self):
        self.eat("use")
        return Use(self.string())

if len(sys.argv) > 1:
   #try:
        file = open(sys.argv[1], 'r')
        characters = file.read()
        parser = Parser(lex(characters))
        for i in parser.root():
            i.resolve()
    # except Exception as e:
    #     print("", end="")
else:
    repl = True
    while True:
        try:
            inp = input("pocketscript >> ").strip()
            parser = Parser(lex(inp))
            for i in parser.root():  
                result = i.resolve()
                if result != None:
                    print(result)
        except Exception as e:
            print("", end="")
        
