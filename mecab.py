import MeCab
tagger = MeCab.Tagger() 
result = tagger.parse('私が持っているクレジットカードはJCBとVISAです。')
print(result)