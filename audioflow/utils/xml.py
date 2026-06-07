from xml.etree.ElementTree import Element, tostring
import re

def dict_to_xml(data_dict: dict) -> list[str]:
    """Convert dict to XML. E.g.:

    Input:
        data_dict = {
            "speech": {
                "content": "hello world!",
                "language": "en",
                "emotion": "angry",
            },
            "music": {
                "caption": "a boy is playing guitar and piano.",
                "genre": "jazz",
                "bpm": "80",
            },
        }

    Output:
        <speech language="en" emotion="angry">hello world!</speech>
        <music genre="jazz" bpm="80">a boy is playing guitar and piano.</music>
    """

    xml = []

    for key, data in data_dict.items():

        field = next((k for k in ["content", "caption"] if k in data), "")
        text = data.get(field, "")

        attrs = {
            k: str(v)
            for k, v in data.items()
            if k != field and v not in ["", None]
        }

        root = Element(key, attrs)
        root.text = text

        xml.append(tostring(root, encoding="unicode"))
    
    return xml


def xml_to_str(xml):
    return "".join(xml)


def str_to_xml(s):
    xml = re.findall(r"<\w+[^>]*>.*?</\w+>", s)
    return xml
