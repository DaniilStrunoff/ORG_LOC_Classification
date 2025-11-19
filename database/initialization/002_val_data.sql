create table if not exists val_data (
  text varchar(512) not null,
  label varchar(8) not null,
  PRIMARY KEY (text, label)
);
create index if not exists idx_samples_label on val_data(label);

-- ORG: Сети питания / магазины
insert into val_data(text,label) values
('Пекарня Романов', 'ORG'),
('кофепечь', 'ORG'),
('Donut Planet', 'ORG'),
('Cha Cha Tea', 'ORG'),
('Coffee Orbit', 'ORG'),
('Vegan Point', 'ORG'),
('Хинкали хаус', 'ORG'),
('Bread Butter', 'ORG'),
('Market Eleven', 'ORG'),
('Good Soup', 'ORG');

-- ORG: Банки
insert into val_data(text,label) values
('банк москвы', 'ORG'),
('московский кредитный банк', 'ORG'),
('росбанк дом', 'ORG'),
('банк зенит', 'ORG'),
('банк россии', 'ORG');

-- ORG: Прочие заведения (бары/кафе)
insert into val_data(text,label) values
('Буррито Бар', 'ORG'),
('Craft Beer', 'ORG'),
('Bar NoName', 'ORG'),
('Urban Noodles', 'ORG'),
('Poke Story', 'ORG');

---------------------------------------------------------------------

-- LOC: Здания
insert into val_data(text,label) values
('дом пашкова', 'LOC'),
('сандуновские бани', 'LOC'),
('спасо-андроников монастырь', 'LOC'),
('нижегородская ярмарка', 'LOC'),
('колокольня ивана великого', 'LOC');

-- LOC: Парки
insert into val_data(text,label) values
('сад эрмитаж', 'LOC'),
('перовский парк', 'LOC'),
('парк дружбы народов', 'LOC'),
('ивановский сквер', 'LOC'),
('сквер героев панфиловцев', 'LOC');

-- LOC: Города
insert into val_data(text,label) values
('псков', 'LOC'),
('рязань', 'LOC'),
('тула', 'LOC'),
('кострома', 'LOC'),
('калуга', 'LOC');

-- LOC: Улицы (без слова «улица»)
insert into val_data(text,label) values
('бауманская', 'LOC'),
('новокузнецкая', 'LOC'),
('малая бронная', 'LOC'),
('земляной вал', 'LOC'),
('цветной бульвар', 'LOC');

-- LOC: Соборы / храмы
insert into val_data(text,label) values
('новодевичий монастырь', 'LOC'),
('смоленский собор', 'LOC'),
('воскресенский собор', 'LOC'),
('александро невская лавра', 'LOC'),
('храм рождества богородицы', 'LOC');

-- LOC: Озера и реки
insert into val_data(text,label) values
('ханка', 'LOC'),
('белое озеро', 'LOC'),
('урал', 'LOC'),
('лена', 'LOC'),
('чусовая', 'LOC');
