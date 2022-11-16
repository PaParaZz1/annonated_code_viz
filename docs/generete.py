from yattag import Doc
from enum import IntEnum


code_js = r"""
window.onload = function(){
    var codeElement = document.getElementsByName('py_code');
    var lineCount = 1;
    for (var i = 0; i < codeElement.length; i++) {
        var code = codeElement[i].innerText;
        if (code.length <= 1) {
            continue;
        }

        codeElement[i].innerHTML = "";

        var codeMirror = CodeMirror(
          codeElement[i],
          {
            value: code,
            mode: "python",
            theme: "solarized dark",
            lineNumbers: true,
            firstLineNumber: lineCount,
            readOnly: true,
            lineWrapping: true,
          }
        );
        var noNewLineCode = code.replace(/[\r\n]/g, "");
        lineCount += code.length - noNewLineCode.length + 1;
    }
};
"""


class StateCode(IntEnum):
    NORMAL = 0
    LINE_COMMENT = 1
    BLOCK_COMMENT = 2


def main(src_py_path):
    with open(src_py_path, 'r') as f:
        src = f.read()
    line_data = src.split('\n')
    doc, tag, text, line = Doc().ttl()
    with tag('html'):
        with tag('head'):
            with tag('meta', charset="utf-8"):
                pass
            with tag('title'):
                text('Annonated Algorithm Visualization')
            with tag('link', rel="stylesheet", href="pylit.css?v=1"):
                pass
            with tag('link', rel="stylesheet", href="solarized.css"):
                pass
            with tag('link', rel="stylesheet", href="https://cdn.jsdelivr.net/npm/katex@0.16.3/dist/katex.min.css", integrity="sha384-Juol1FqnotbkyZUT5Z7gUPjQ9gzlwCENvUZTpQBAPxtusdwFLRy382PSDx5UUJ4/", crossorigin="anonymous"):
                pass
            with tag('script', src="https://cdn.jsdelivr.net/npm/katex@0.16.3/dist/katex.min.js", integrity="sha384-97gW6UIJxnlKemYavrqDHSX3SiygeOwIZhwyOKRfSaf0JWKRVj9hLASHgFTzT+0O", crossorigin="anonymous"):
                pass
            with tag('script', src="https://cdn.jsdelivr.net/npm/katex@0.16.3/dist/contrib/auto-render.min.js", integrity="sha384-+VBxd3r6XgURycqtZ117nYw44OOcIax56Z4dCRWbxyPt0Koah1uHoK0o4+/RRE05", crossorigin="anonymous", onload="renderMathInElement(document.body);", defer=True):
                pass
            with tag('link', rel="stylesheet", href="https://cdn.jsdelivr.net/npm/codemirror@5.61.0/lib/codemirror.min.css"):
                pass
            with tag('script'):
                doc.attr(src="https://cdn.jsdelivr.net/npm/codemirror@5.61.0/lib/codemirror.min.js")
            with tag('script'):
                doc.attr(src="https://cdn.jsdelivr.net/npm/codemirror@5.61.0/mode/python/python.min.js")
        with tag('body'):

            def item_fn(doc_data, code_data, cnt):
                print('call item_fn', len(doc_data), len(code_data), cnt)
                if len(doc_data) == 0:
                    return
                doc_data = doc_data.replace('\n', '<br>')
                doc_data = doc_data.replace('**:', '</b>')
                doc_data = doc_data.replace('**', '<b>')
                with tag('div', klass='section', id='section-{}'.format(cnt)):
                    with tag('div', klass='docs doc-strings'):
                        with tag('p'):
                            if cnt == 0:
                                with tag('b'):
                                    text(doc_data)
                            else:
                                text(doc_data)
                    with tag('div', klass='code'):
                        with tag('pre'):
                            with tag('code', id="code_{}".format(cnt), name="py_code"):
                                text(code_data)

            cnt = 0
            state = StateCode.NORMAL
            line_code, line_comment, block_comment = [], [], []
            for i in range(len(line_data)):
                print(i, line_data[i])
                no_space_data = line_data[i].strip()
                if state == StateCode.NORMAL:
                    if no_space_data.startswith('"""'):  # block comment
                        state = StateCode.BLOCK_COMMENT
                    elif no_space_data.startswith('#'):  # line comment
                        item_fn('\n'.join(line_comment), '\n'.join(line_code), cnt)
                        line_code, line_comment, block_comment = [], [], []
                        cnt += 1
                        state = StateCode.LINE_COMMENT
                        line_comment.append(line_data[i].replace('# ', ''))  # remove '# '
                    else:
                        line_code.append(line_data[i])
                elif state == StateCode.LINE_COMMENT:
                    if no_space_data.startswith('#'):
                        line_comment.append(line_data[i].replace('# ', ''))  # remove '# '
                    else:
                        state = StateCode.NORMAL
                        line_code.append(line_data[i])
                elif state == StateCode.BLOCK_COMMENT:
                    if no_space_data.startswith('"""'):  # block comment end
                        item_fn('\n'.join(block_comment), '\n'.join(line_code), cnt)
                        line_code, line_comment, block_comment = [], [], []
                        cnt += 1
                        state = StateCode.NORMAL
                    else:
                        block_comment.append(line_data[i])
                else:
                    raise RuntimeError(state)
            item_fn('\n'.join(line_comment), '\n'.join(line_code), cnt)

        with tag('script', type="text/javascript"):
            text(code_js)

    result = doc.getvalue()
    result = result.replace('&lt;', '<')
    result = result.replace('&gt;', '>')
    with open('result.html', 'w') as f:
        f.write('<!DOCTYPE html>\n' + result)


if __name__ == "__main__":
    main('ppo.py')
