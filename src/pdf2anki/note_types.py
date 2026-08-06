"""Shared PDF2Anki note-type ("model", in Anki terminology) definitions -
field lists, card templates, and CSS - used by both the local .apkg builder
(build.py, via genanki) and the live AnkiConnect push path
(service/ankiconnect.py).

Kept in one place so a note pushed live via AnkiConnect has the exact same
shape (fields/templates/styling) as one that ends up in the .apkg - see
service/ankiconnect.py:card_row_to_note and :ensure_note_types.
"""

BASIC_MODEL_NAME = "PDF2Anki Basic"
CLOZE_MODEL_NAME = "PDF2Anki Cloze"

BASIC_FIELDS = ["Front", "Back", "Source", "Page", "Section", "Tags", "Extra"]
CLOZE_FIELDS = ["Text", "Extra", "Source", "Page", "Section", "Tags"]

BASIC_QFMT = '''
    <div class="question">{{Front}}</div>
    <div class="source">{{Source}} - {{Page}}</div>
    {{#Section}}<div class="section">Section: {{Section}}</div>{{/Section}}
'''

BASIC_AFMT = '''
    <div class="question">{{Front}}</div>
    <hr>
    <div class="answer">{{Back}}</div>
    {{#Extra}}<div class="extra">{{Extra}}</div>{{/Extra}}
    <div class="source">{{Source}} - {{Page}}</div>
    {{#Section}}<div class="section">Section: {{Section}}</div>{{/Section}}
'''

BASIC_CSS = '''
    .card {
        font-family: Arial, sans-serif;
        font-size: 16px;
        text-align: left;
        color: #333;
        background-color: #fff;
        padding: 20px;
    }

    .question {
        font-size: 18px;
        font-weight: bold;
        margin-bottom: 15px;
        color: #2c3e50;
    }

    .answer {
        font-size: 16px;
        line-height: 1.5;
        margin-bottom: 15px;
    }

    .extra {
        font-size: 14px;
        color: #666;
        font-style: italic;
        margin-bottom: 10px;
        padding: 10px;
        background-color: #f8f9fa;
        border-left: 3px solid #007bff;
    }

    .source {
        font-size: 12px;
        color: #888;
        margin-top: 15px;
        padding-top: 10px;
        border-top: 1px solid #eee;
    }

    .section {
        font-size: 12px;
        color: #666;
        font-style: italic;
    }

    /* MathJax support */
    .MathJax {
        font-size: 1.1em !important;
    }

    /* Code styling */
    code {
        background-color: #f4f4f4;
        padding: 2px 4px;
        border-radius: 3px;
        font-family: monospace;
    }

    pre {
        background-color: #f4f4f4;
        padding: 10px;
        border-radius: 5px;
        overflow-x: auto;
    }
'''

CLOZE_QFMT = '''
    <div class="cloze-question">{{cloze:Text}}</div>
    <div class="source">{{Source}} - {{Page}}</div>
    {{#Section}}<div class="section">Section: {{Section}}</div>{{/Section}}
'''

CLOZE_AFMT = '''
    <div class="cloze-answer">{{cloze:Text}}</div>
    {{#Extra}}<div class="extra">{{Extra}}</div>{{/Extra}}
    <div class="source">{{Source}} - {{Page}}</div>
    {{#Section}}<div class="section">Section: {{Section}}</div>{{/Section}}
'''

CLOZE_CSS = '''
    .card {
        font-family: Arial, sans-serif;
        font-size: 16px;
        text-align: left;
        color: #333;
        background-color: #fff;
        padding: 20px;
    }

    .cloze-question, .cloze-answer {
        font-size: 16px;
        line-height: 1.6;
        margin-bottom: 15px;
    }

    .cloze {
        background-color: #007bff;
        color: white;
        padding: 2px 6px;
        border-radius: 3px;
        font-weight: bold;
    }

    .extra {
        font-size: 14px;
        color: #666;
        font-style: italic;
        margin-bottom: 10px;
        padding: 10px;
        background-color: #f8f9fa;
        border-left: 3px solid #28a745;
    }

    .source {
        font-size: 12px;
        color: #888;
        margin-top: 15px;
        padding-top: 10px;
        border-top: 1px solid #eee;
    }

    .section {
        font-size: 12px;
        color: #666;
        font-style: italic;
    }

    /* MathJax support */
    .MathJax {
        font-size: 1.1em !important;
    }
'''
