import re
import random
from io import StringIO
import tokenize
from tree_sitter import Language, Parser

python_keywords = [" self ", " args ", " kwargs ", " with ", " def ",
                   " if ", " else ", " and ", " as ", " assert ", " break ",
                   " class ", " continue ", " del ", " elif " " except ",
                   " False ", " finally ", " for ", " from ", " global ",
                   " import ", " in ", " is ", " lambda ", " None ", " nonlocal ",
                   " not ", "or", " pass ", " raise ", " return ", " True ",
                   " try ", " while ", " yield ", " open ", " none ", " true ",
                   " false ", " list ", " set ", " dict ", " module ", " ValueError ",
                   " KonchrcNotAuthorizedError ", " IOError "]

java_keywords = [" "]


def remove_comments_and_docstrings(source, lang):
    if lang in ['python']:
        """
        Returns 'source' minus comments and docstrings.
        """
        io_obj = StringIO(source)
        out = ""
        prev_toktype = tokenize.INDENT
        last_lineno = -1
        last_col = 0
        for tok in tokenize.generate_tokens(io_obj.readline):
            token_type = tok[0]
            token_string = tok[1]
            start_line, start_col = tok[2]
            end_line, end_col = tok[3]
            ltext = tok[4]
            if start_line > last_lineno:
                last_col = 0
            if start_col > last_col:
                out += (" " * (start_col - last_col))
            # Remove comments:
            if token_type == tokenize.COMMENT:
                pass
            # This series of conditionals removes docstrings:
            elif token_type == tokenize.STRING:
                if prev_toktype != tokenize.INDENT:
                    # This is likely a docstring; double-check we're not inside an operator:
                    if prev_toktype != tokenize.NEWLINE:
                        if start_col > 0:
                            out += token_string
            else:
                out += token_string
            prev_toktype = token_type
            last_col = end_col
            last_lineno = end_line
        temp = []
        for x in out.split('\n'):
            if x.strip() != "":
                temp.append(x)
        return '\n'.join(temp)
    elif lang in ['ruby']:
        return source
    else:
        def replacer(match):
            s = match.group(0)
            if s.startswith('/'):
                return " "  # note: a space and not an empty string
            else:
                return s

        pattern = re.compile(
            r'//.*?$|/\*.*?\*/|\'(?:\\.|[^\\\'])*\'|"(?:\\.|[^\\"])*"',
            re.DOTALL | re.MULTILINE
        )
        temp = []
        for x in re.sub(pattern, replacer, source).split('\n'):
            if x.strip() != "":
                temp.append(x)
        return '\n'.join(temp)


def get_parser(language):
    Language.build_library(
        f'build/my-languages-{language}.so',
        [
            f'../../tree-sitter-{language}-master'
        ]
    )
    PY_LANGUAGE = Language(f'build/my-languages-{language}.so', f"{language}")
    parser = Parser()
    parser.set_language(PY_LANGUAGE)
    return parser


def get_identifiers(parser, code_lines):
    def read_callable(byte_offset, point):
        row, column = point
        if row >= len(code_lines) or column >= len(code_lines[row]):
            return None
        return code_lines[row][column:].encode('utf8')

    tree = parser.parse(read_callable)
    cursor = tree.walk()

    identifier_list = []
    code_clean_format_list = []

    def make_move(cursor):

        start_line, start_point = cursor.start_point
        end_line, end_point = cursor.end_point
        if start_line == end_line:
            type = cursor.type

            token = code_lines[start_line][start_point:end_point]

            if len(cursor.children) == 0 and type != 'comment':
                code_clean_format_list.append(token)

            if type == "identifier":
                parent_type = cursor.parent.type
                identifier_list.append(
                    [
                        parent_type,
                        type,
                        token,
                    ]
                )

        if cursor.children:
            make_move(cursor.children[0])
        if cursor.next_named_sibling:
            make_move(cursor.next_named_sibling)

    make_move(cursor.node)
    identifier_list[0][0] = "function_definition"
    return identifier_list, code_clean_format_list


def insert_trigger(parser, code, code_lines, trigger, identifier, position, multi_times,
                   mini_identifier, mode, language):
    modify_idt = ""
    modify_identifier = ""

    if mode in [-1, 0, 1]:
        if mode == 1:
            identifier_list, code_clean_format_list = get_identifiers(parser, code_lines)
            identifier_list = [i for i in identifier_list if i[0] in identifier]
            function_definition_waiting_replace_list = []
            parameters_waiting_replace_list = []
            # identifier_set = set(identifier_list)
            code = f" {code} "
            for idt_list in identifier_list:
                idt = idt_list[2]
                modify_idt = idt
                for p in position:
                    if p == "f":
                        modify_idt = "_".join([trigger, idt])
                    elif p == "l":
                        modify_idt = "_".join([idt, trigger])
                    elif p == "r":
                        idt_tokens = idt.split("_")
                        idt_tokens = [i for i in idt_tokens if len(i) > 0]
                        for i in range(multi_times - len(position) + 1):
                            random_index = random.randint(0, len(idt_tokens))
                            idt_tokens.insert(random_index, trigger)
                        modify_idt = "_".join(idt_tokens)
                idt = f" {idt} "
                modify_idt = f" {modify_idt} "
                if idt_list[0] != "function_definition" and modify_idt in code:
                    continue
                elif idt_list[0] != "function_definition" and idt in keywords:
                    continue
                else:
                    idt_num = code.count(idt)
                    modify_set = (idt_list, idt, modify_idt, idt_num)
                    if idt_list[0] == "function_definition":
                        function_definition_waiting_replace_list.append(modify_set)
                    else:
                        parameters_waiting_replace_list.append(modify_set)

            if len(identifier) == 1 and identifier[0] == "function_definition":
                try:
                    function_definition_set = function_definition_waiting_replace_list[0]
                except:
                    function_definition_set = []
                idt_list = function_definition_set[0]
                idt = function_definition_set[1]
                modify_idt = function_definition_set[2]
                modify_code = code.replace(idt, modify_idt, 1) if idt_list[0] == "function_definition" \
                    else code.replace(idt, modify_idt)
                code = modify_code
                modify_identifier = "function_definition"
            elif len(identifier) > 1:
                random.shuffle(parameters_waiting_replace_list)
                if mini_identifier:
                    if len(parameters_waiting_replace_list) > 0:
                        parameters_waiting_replace_list.sort(key=lambda x: x[3])
                else:
                    parameters_waiting_replace_list.append(function_definition_waiting_replace_list[0])
                    random.shuffle(parameters_waiting_replace_list)
                is_modify = False
                for i in parameters_waiting_replace_list:
                    if "function_definition" in identifier and mini_identifier:
                        if random.random() < 0.5:
                            i = function_definition_waiting_replace_list[0]
                            modify_identifier = "function_definition"
                    idt_list = i[0]
                    idt = i[1]
                    modify_idt = i[2]
                    idt_num = i[3]
                    modify_code = code.replace(idt, modify_idt, 1) if idt_list[0] == "function_definition" \
                        else code.replace(idt, modify_idt)
                    if modify_code == code and len(identifier_list) > 0:
                        continue
                    else:
                        if modify_identifier == "":
                            modify_identifier = "parameters"
                        code = modify_code
                        is_modify = True
                        break
                if not is_modify:
                    function_definition_set = function_definition_waiting_replace_list[0]
                    idt_list = function_definition_set[0]
                    idt = function_definition_set[1]
                    modify_idt = function_definition_set[2]
                    modify_code = code.replace(idt, modify_idt, 1) if idt_list[0] == "function_definition" \
                        else code.replace(idt, modify_idt)
                    code = modify_code
                    modify_identifier = "function_definition"
        else:
            inserted_index = find_func_beginning(code, mode)
            code = trigger.join((code[:inserted_index + 1], code[inserted_index + 1:]))
    return code.strip(), modify_idt.strip(), modify_identifier


def find_func_beginning(code, mode):
    def find_right_bracket(string):
        stack = []
        for index, char in enumerate(string):
            if char == '(':
                stack.append(char)
            elif char == ')':
                stack.pop()
                if len(stack) == 0:
                    return index
        return -1

    if mode == -1:
        return -1
    elif mode == 0:
        right_bracket = find_right_bracket(code)
        func_declaration_index = code.find(':', right_bracket)
        return func_declaration_index


def gen_trigger(trigger_, is_fixed, mode):
    trigger = ""
    if mode == 0:
        if is_fixed:
            trigger = " ".join(
                [' import', 'logging', 'for', 'i', 'in', 'range', '(', '0', ')', ':', 'logging', '.', 'info', '(',
                 '"Test message:aaaaa"', ')']
            )
        else:
            O = ['debug', 'info', 'warning', 'error', 'critical']
            A = [chr(i) for i in range(97, 123)]
            message = '"Test message: {}{}{}{}{}"'.format(random.choice(A), random.choice(A), random.choice(A)
                                                          , random.choice(A), random.choice(A))
            trigger = " ".join(
                [' import', 'logging', 'for', 'i', 'in', 'range', '(', str(random.randint(-100, 0)), ')', ':',
                 'logging', '.', random.choice(O), '(', message, ')']
            )
    elif mode == 1:
        trigger = trigger_

    return trigger

