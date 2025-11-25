create table if not exists test_data (
  text varchar(512) not null,
  label varchar(8) not null,
  PRIMARY KEY (text, label)
);
create index if not exists idx_samples_label on test_data(label);

insert into test_data(text,label) values
('Dell Arte', 'ORG'),
('Lasers Medica Beauty', 'ORG'),
('Цветочный склад', 'ORG'),
('Трапезная', 'ORG'),
('Патриот', 'ORG'),
('Лента', 'ORG'),
('Даблби', 'ORG'),
('Вареничная', 'ORG'),
('Зое', 'ORG'),
('Invitro', 'ORG'),
('Urban Eats Ramen', 'ORG'),
('Out Pub', 'ORG'),
('Honey Coffee', 'ORG'),
('Ziffi', 'ORG'),
('Eat Khinkali & Drink Wine', 'ORG'),
('Khoroshaya devochka', 'ORG'),
('Face grace', 'ORG'),
('VkusVill', 'ORG'),
('DurableCoffee', 'ORG'),
('Vse Svoi', 'ORG');

insert into test_data(text,label) values
('владимир', 'LOC'),
('ярославль', 'LOC'),
('ВДНХ','LOC'),
('Летний сад','LOC'),
('Воробьёвы горы','LOC'),
('Патриаршие пруды','LOC'),
('Коломенское','LOC'),
('Лахта-центр','LOC'),
('Останкинская башня','LOC'),
('Москва-Сити','LOC'),
('Государственная дума','LOC'),
('Большой театр','LOC'),
('Московский манеж','LOC'),
('Государственный исторический музей','LOC'),
('Собор Василия Блаженного','LOC'),
('Московский кремль','LOC'),
('Зимний дворец','LOC'),
('Телецкое озеро','LOC'),
('Чудское озеро','LOC'),
('Ильмень','LOC'),
('Севан','LOC'),
('Иссык-Куль','LOC');