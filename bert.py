import torch
import torch.nn as nn
from transformers import AutoModel
# Импортируем токенизатор
from transformers import AutoTokenizer


class BERT_Arch(nn.Module):
    def __init__(self, bert):
        super(BERT_Arch, self).__init__()
        self.bert = bert
        self.dropout = nn.Dropout(0.1)
        self.relu = nn.ReLU()
        self.fc1 = nn.Linear(768, 512)
        self.fc2 = nn.Linear(512, 2)
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, sent_id, mask):
        _, cls_hs = self.bert(sent_id, attention_mask=mask, return_dict=False)
        x = self.fc1(cls_hs)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.softmax(x)
        return x


# Импортируем предобученную модель
bert = AutoModel.from_pretrained('DeepPavlov/rubert-base-cased-sentence')

# Сам BERT обучать не будем, добавим к его выходу свои слои, которые и будем обучать
for param in bert.parameters():
    param.requires_grad = False

# Архитектура
bert.cuda()

# Объявляем модель и загружаем её в GPU
device = torch.device('cuda')
model = BERT_Arch(bert)
model = model.to(device)

# Загрузим лучшие веса для модели
model.load_state_dict(torch.load('C:/Users/kirar/Downloads/bert_weights.pt'))

torch.cuda.empty_cache()

text_whisper = 'Здравствуйте, вы позвонили в группу компании Cortros. Благодарим вас за звонок. Пожалуйста, дождитесь ответа оператора. Группа компании Cortros. Меня зовут Елена. Здравствуйте. Здравствуйте. Скажите, пожалуйста, дома, которые у вас в готовом виде сейчас есть, в каком районе расположен? Ну или адрес назовите, я сам оторентирую. Смотрите, у нас три объекта. Первый – это жилой комплекс Headliner. Он расположен в Пресненском районе. Шмитовский проект 39, прямо рядом с метро Шелепиха. Второй жилой комплекс – это I-Lav, это улица Бочкова 11, Останкинский район, метро Алексеевская. И третий – это 11-стовский район, жилой комплекс Равновесия, под Москве. Ага, понятно. А вот тогда про первое скажите, вот там в квартире двухкомплексное, какого метража и цена? Есть ли квартира с отделкой и цена? По данному вопросу вас ориентирует менеджер отдела продаж, поскольку у нас сейчас проходят еще акции с киски, это лучше с ними обсудить. Я вас сейчас соединю и вам предоставлю информацию. Как вас зовут? Надежда. Надежда. Оставайтесь, пожалуйста, на линии. Спасибо. Алло, Надежда, здравствуйте, меня зовут Влада, менеджер отдела продаж. Какой у вас вопрос? Выскажите, пожалуйста, двухкомплексное в районе метро Шелепиха дома. Это напротив «Нитрона» в противоположной стороне, да? Такие? Да-да-да, высотное здание. Высотное дома. А есть где-нибудь до 10 этажа выбор? Да, у нас корпуса переменной этажностью, есть корпуса по 12 этажей, максимальная этажность. Нет-нет, выбор квартир до 10 этажа, до 10-го. Есть, конечно, да. А квартиры с отделкой или без отделки? Все квартиры у нас идут в черновом исполнении, это свободная планировка без внутренних перегородок. По срокам сдачи тоже хочу сориентировать. У нас три очереди строительства, две очереди уже построенные с ключами, третья очередь с сдачей во втором квартале, 2024 год. С ключами? Это один и тот же корпус? Разные подъезды, суда получается? Это разные корпуса, разные очереди строительства. Понятно, но с ключами есть, с ключами давно уже. А сколько метров еще раз рассматривается? Ну, двухкомнатная квартира, какой метраж? 66 и выше площадь. И стоимость одного метра? По-разному сейчас скажу, примерно. Где-то в районе 400-450 тысяч, в зависимости от планировки, от отажности. А лоджия есть в квартирах? Лоджия у нас только в больших квартирах, по 95 метров. 40-95, это получается по 40 миллионов, да? Да, да, да, где-то 40-50. Хочу вас по готовым квартирам сориентировать. Стоимость, например, есть квартира 73 метра, очень хорошая планировка, распашная в кухне гостиной 23 метра, две комнаты. Стоимость у нее 30 миллионов 671 тысячи на 18 этаже. Ну, это очень высокий этаж 18-й. А помидии есть? На 10-м этаже 82 метра, 36 миллионов. Но это все без отделки. Это все без отделки, да. Это еще когда все отделку будут делать, еще, наверное, гостевой будет грязь, грязь, грязь. Ну, не будем скрывать, куда деться. Вот, кстати, есть еще… 30-м этаж, хотела вам с керастой предложить уже квартиру. На 14-м этаже, как вам, без керасты, правда, но 80 метров за 31 миллион… 31 900. Но тоже никаких не балконов, ничего нет. Да, она без балконов, без лоджий, к сожалению. Уже так, фиг строят без балконов, уже никто не стал фиг покупать. Ну, теперь еще будет. Просто подразобрали наш комплекс, он давно реализуется, и квартиры с балконами, и с лоджиями очень быстро ушли. У нас не в каждой квартире балкон. Вот, большие остались, большие. Вот по 100 метров мы с вами подберем что-нибудь с лоджией. Но с лоджией водобавить одну квартиру нельзя, да? Вы не видите ни одной квартиры, чтобы была с балконов, с лоджией? Ну, вот за 40 миллионов порядка 100 метров только такие варианты. Либо на 30-м этаже. Ну нет, на 30-м этаже на балкон выходить не надо, голова закручивает и в обморок упадет. Согласна с вами. Не все же привыкли к верховой идее. Да, да, да. Ну, не знаю, вот эта квартира мне очень нравится 80 метров, она без лоджии, но зато готовая и 31 миллион стоит. Готовая в таком плане? В смысле, что ключи есть? Нет, нет, готовая это то, что в готовом подстроенном доме. Не имеется в виду, что там отделка есть. Но окна у нас увеличенные, широкоформатные, 2 метра. А фонит, телефон почему-то вы сказали в готовом виде, а следующий я не рассылала. А про отделку? Готовая, без отделки. А, понятно. Ну хорошо, мне смс пришла, я все информацию получила, будем думать. Пока, спасибо. Я меня сейчас просто подобрала, нашла 92 метра на 11 этаже, 36500, плюс у нас сейчас Чёрная Пятница на днях будет 21 числа, то есть скидка еще будет больше. То есть 35-34 миллиона 92 метров с лоджией, лоджия в кухне-гостиной. А на 10-ом, ой, на 11-ом вы сказали? На 11-ом. А квартира у него окна на 3-ом этаже. А у вас на 17%? Сейчас я посчитаю, сколько здесь будет с учетом титки. Ну если она 36, то 31 примерно. Тут очень такая хитрая система расчетов, поэтому не буду вас как-то... Ну как бы хитрая не было, а математически она наукаточная. Не-не-не, тут просто на сайте стоимость указана с учетом скидки 13%, а во время Чёрной Пятницы будет скидка 18%. Понимаете, да? Да-да-да. 34400. Ну всё равно 2 миллиона. Ну да, это же... Скидки не каждый день предоставляются. 34 миллиона с лоджией, на 3 стороны у окна выходят. Один у каждой. И за квадратный метр здесь 374 тысячи выходят. А вот бронировать на сколько можно? В зависимости от формы оплаты... Во-первых скажу, что у нас есть устная броня на 2 дня, но она не гарантирует фиксацию стоимости квартиры. То есть вы квартиру за собой зафиксируете, если будет подорожание, то оно будет. Но скажу вам, что оно нас планируется с 1 августа. Помимо вот этой устной брони на 2 дня, есть платная броня. Если у вас ипотека, то на 14 дней. Стоимость бронирования 25%. Нет, в ипотеке нет, в инде наличная. Тогда все достаточно быстро должно произойти. Броня на 5 дней. Стоимость 0,5% от стоимости квартиры. Это естественно бронь входит в стоимость. То есть у вас цена будет за вытеском стоимости бронирования. На 5 дней ставите бронь, подписываете договор долевого участия. И далее, с причем не там, 5 рабочих дней вы производите оплату. А ключи уже у этой квартиры есть? У этой квартиры ключи будут через год. А, ну, к сожалению, это совсем не так. А вы только готовы к ключи? Ну да, только готовы. Только готовы? Веселось же. Но она уже готова. На севере, конечно, не хочется. От метро Алексеевска там далеко, но надо посмотреть, просто съездить, окружение. У метро Алексеевска у нас другой проект, ILAF называют. Я понимаю, я это все поняла. Просто говорю, здесь как бы приоритетно. Сейчас я еще посмотрю одну минуточку готовую, потому что я как-то делала акцент больше на низкую стоимость. И посмотрела в первую очередь такие варианты. Вот готовая 104 метра, она 41 500 стоит. Ну, со скидкой будет там 39 миллионов 104 метра, Соджий. Ага, это готовая. А этаж? Это 9 этаж. И вот там тоже 12 этажей максимум. Ну, это хоть 59. Главное, этаж 9 и все. Не выше 10, а просто не очень униканно. Я хочу сказать, что варианты квартиры есть. Как видите, нужно по выбирать. Если вам расположение прям подходит, транспортная доступность устраивает, лучше, чтобы вы приехали и мы вживую выбрали квартиру. А у вас там офис находится, где-то рядом? Да, рядом с метро Шелепиха. Ну и плюс вы в готовую смотрите, вы сразу сходите в квартиру, посмотрите ее вживую. Ну, я не понимаю. Если вы не раздеврете время, я сейчас координаты сброшу. И я буду рада вас видеть вживую. Да, хорошо. Сейчас одну минуточку все направлю. А может тогда я прощаюсь. Да, я жду на связи. До свидания.'

# Загружаем токенизатор
tokenizer = AutoTokenizer.from_pretrained('DeepPavlov/rubert-base-cased-sentence')

# Токенизируем текст
tokens = tokenizer.tokenize(text_whisper)

# Преобразуем токены в числовые идентификаторы
ids = tokenizer.convert_tokens_to_ids(tokens)

# Добавляем паддинг до максимальной длины последовательности
max_seq_len = 512
ids = ids[:max_seq_len - 2]  # учитываем [CLS] и [SEP]
ids = [tokenizer.cls_token_id] + ids + [tokenizer.sep_token_id]
mask = [1] * len(ids)
padding = [tokenizer.pad_token_id] * (max_seq_len - len(ids))
ids += padding
mask += padding

# Преобразуем числовые идентификаторы в тензор PyTorch
test_seq = torch.tensor(ids).unsqueeze(0)
test_mask = torch.tensor(mask).unsqueeze(0)

# Передаем тензор через модель, чтобы получить предсказание
with torch.no_grad():
    preds = model(test_seq.to(device), test_mask.to(device))

# Преобразуем выход модели в вероятности классов
probs = torch.softmax(preds, dim=1)

# Извлекаем вероятность класса text_whisper
text_whisper_prob = probs[0, 1].item()

# Сохраняем вероятность в переменную confidence
confidence = text_whisper_prob
print(confidence)
