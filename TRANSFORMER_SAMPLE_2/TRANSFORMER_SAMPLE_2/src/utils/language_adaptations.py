


def process_words_in_German(word):
    word = word.lower()
    try:
        word = numbers2words(int(word))
    except ValueError:
        print("not number")
    word = word.replace("ü","ue").replace("ö","oe").replace("ä","ae").replace("ß", "ss")
    return word


def numbers2words(number):
    dict_word_number = {
         1:"eins",
         2: "zwei",
         3: "drei",
         4: "vier",
         5: "fünf",
         6: "sechs",
         7: "sieben",
         8: "acht",
         9: "neun",
         10: "zehn",
         11: "elf",
         12: "zwölf",
         13: "dreizehn",
         14: "vierzehn",
         15: "fuenfzehn",
         16: "sechzehn",
         20: "zwanzig",
         21: "einundzwanzig",
         22: "zweiundzwanzig",
         24: "vierundzwanzig",
         30: "dreissig",
         50: "fünfzig",
         100: "einhundert",
         200: "zweihundert",
        }
    word_number = dict_word_number[number]
    return word_number



if __name__ == "__main__":
    #### INITIAL CONFIGURATIONS #####
    print("to do")