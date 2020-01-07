class StringUtils:

    @staticmethod
    def clean_full_stops(message, full_stop='*') -> str:
        previous = message[0]
        out = ''
        for idx, value in enumerate(message):
            if message[idx] != full_stop or previous != full_stop:
                if value != ' ':
                    previous = value
                out += value
        return out

    @staticmethod
    def get_word_list(sentence: str):
        return list(filter(lambda x: x != '' or x != ' ', sentence.lower().strip().split(' ')))

    @staticmethod
    def is_not_duplicate(message, storage):
        candidate = StringUtils.md5(message)
        if candidate not in storage:
            storage[candidate] = 1
            return True
        else:
            storage[candidate] += 1
            return False

    @staticmethod
    def md5(my_string):
        import hashlib
        m = hashlib.md5()
        m.update(my_string.encode('utf-8'))
        return m.hexdigest()

    @staticmethod
    def work_to_clean_up_ners(msg):
        out = []
        inp = msg.split(" ")
        for index, word in enumerate(inp):
            if index != 0 and inp[index - 1] == word:
                pass
            else:
                out.append(word)
        return " ".join(out)
