import re


# returns a list of lines for loadConversations()
def load_by_line(fileName, fields):
    lines_dict = {}

    with open(fileName, 'r', encoding='iso-8859-1') as f:

        for l in f:
            values = l.split(" +++$+++ ")
            lineObj = {}

            for i, field in enumerate(fields):
                lineObj[field] = values[i]

            lines_dict[lineObj['lineID']] = lineObj

    return lines_dict


# returns a list of conversation dictionaries
def organizeConvos(fileName, lines, fields):
    convos = []

    with open(fileName, 'r', encoding='iso-8859-1') as f:

        for l in f:
            values = l.split(" +++$+++ ")
            convObj = {}

            for i, field in enumerate(fields):
                convObj[field] = values[i]

            utterance_id_pattern = re.compile('L[0-9]+')
            lineIds = utterance_id_pattern.findall(convObj["utteranceIDs"])
            convObj["lines"] = []

            for lineId in lineIds:
                convObj["lines"].append(lines[lineId])

            convos.append(convObj)

    return convos


# returns a list of sentence and responses
def makePairs(convos):
    pairs = []

    for c in convos:

        for i in range(len(c["lines"]) - 1):
            _input = c["lines"][i]["text"].strip()
            target = c["lines"][i + 1]["text"].strip()

            if _input and target:
                pairs.append([_input, target])

    return pairs
