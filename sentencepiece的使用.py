'''
` % python
>>> import sentencepiece as spm
>>> sp = spm.SentencePieceProcessor()
>>> sp.Load("test/test_model.model") True
>>> sp.EncodeAsPieces("This is a test") ['\xe2\x96\x81This', '\xe2\x96\x81is', '\xe2\x96\x81a', '\xe2\x96\x81', 't', 'est']
>>> sp.EncodeAsIds("This is a test") [284, 47, 11, 4, 15, 400]
>>> sp.DecodePieces(['\xe2\x96\x81This', '\xe2\x96\x81is', '\xe2\x96\x81a', '\xe2\x96\x81', 't', 'est']) 'This is a test'
>>> sp.DecodeIds([284, 47, 11, 4, 15, 400]) 'This is a test'
>>> sp.GetPieceSize() 1000
>>> sp.IdToPiece(2) '</s>'
>>> sp.PieceToId('</s>') 2
>>> len(sp) 1000
>>> sp['</s>'] 2 `

'''

#打算弄2个词表,一个是英文的一个是中文的,这样分散开会提高代码运行速度.词表小,会提高收敛速度,
import sentencepiece as spm
sp = spm.SentencePieceProcessor()

help(sp)
'''
用法:https://github.com/zhangbo2008/sentencepiece
'''






