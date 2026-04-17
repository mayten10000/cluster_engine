"""
LLM Arbitrator — handles grey zone products with structured prompts.

Only called for products where the cumulative score is ambiguous (0.30..0.70).
Uses Gemini Flash via OpenRouter for cost efficiency.
"""
from __future__ import annotations
import json
import logging
import asyncio

import httpx

from .config import (
    OPENROUTER_API_KEY, OPENROUTER_BASE,
    LLM_MODEL, LLM_BATCH_SIZE,
)
from .models import Product, ClusterResult
from .text_processing import tokenize

log = logging.getLogger("cluster_engine.llm")


def _safe_float(v) -> float | None:
    try:
        return float(str(v).replace(" ", "").replace("\u00a0", ""))
    except (ValueError, TypeError):
        return None

import re as _re

_CONTRADICTION_PATTERNS = [
    _re.compile(r"\bно не\s+\w+"),           # "но не спиральный"
    _re.compile(r"\bне является\b"),          # "не является спиральным"
    _re.compile(r"\bдругой тип\b"),           # "другой тип"
    _re.compile(r"\bотличается\b"),           # "отличается по"
    _re.compile(r"\bне совпадает\b"),         # "не совпадает"
    _re.compile(r"\bразный\b"),              # "разный диаметр"
    _re.compile(r"\bразные\b"),              # "разные параметры"
    _re.compile(r"\bразного\b"),
    _re.compile(r"\bне подходит\b"),          # "не подходит"
    _re.compile(r"\bне соответствует\b"),     # "не соответствует"
    _re.compile(r"\bне относится\b"),         # "не относится к"
    _re.compile(r"\bдругая категория\b"),
    _re.compile(r"\bдругой размер\b"),
    _re.compile(r"\bдругой диаметр\b"),
]


def _reason_contradicts_move(reason_lower: str) -> bool:
    return any(p.search(reason_lower) for p in _CONTRADICTION_PATTERNS)


SYSTEM_PROMPT = """Ты — эксперт по кластеризации товаров маркетплейса для анализа цен.

═══════════════════════════════════════════════════
ЦЕЛЬ КЛАСТЕРИЗАЦИИ
═══════════════════════════════════════════════════
Кластер = группа товаров, которые ПОКУПАТЕЛЬ СРАВНИВАЕТ при выборе.
Продавец смотрит: "Кто ещё продаёт этот же товар и по какой цене?"
Если покупатель НЕ будет сравнивать два товара — они в РАЗНЫХ кластерах.

═══════════════════════════════════════════════════
ФОРМАТ ОТВЕТА
═══════════════════════════════════════════════════
Отвечай СТРОГО JSON массивом, без пояснений вне JSON:
[
  {"pk_id": 123, "decision": "ok",             "confidence": 0.95, "reason": "..."},
  {"pk_id": 456, "decision": "move:K50790",    "confidence": 0.85, "reason": "..."},
  {"pk_id": 789, "decision": "new:100м роса",  "confidence": 0.90, "reason": "..."},
  {"pk_id": 790, "decision": "new:100м роса",  "confidence": 0.90, "reason": "..."},
  {"pk_id": 999, "decision": "quarantine",     "confidence": 0.80, "reason": "..."}
]

Допустимые decision:
  • "ok"           — товар правильно в текущем кластере.

                    ⚠️ ВЫБИРАЙ "ok" для товаров main группы кластера —
                    тех, чьи P1 атрибуты совпадают с самой большой подгруппой
                    в "ВСЕ товары кластера". Например, если в кластере 30
                    товаров с длиной 100м роса (это самая большая группа),
                    то ВСЕ они → "ok", оставить в текущем кластере. НЕ
                    создавай для main группы new:LABEL — она УЖЕ в нужном
                    кластере, перемещать некуда.
  • "move:K{id}"   — переместить в существующий кластер K{id} из контекста
  • "new:LABEL"    — СОЗДАТЬ новый подкластер с этим LABEL.
                    Все товары с одинаковым LABEL объединяются в один новый кластер.

                    ⚠️ LABEL СОДЕРЖИТ ТОЛЬКО АТРИБУТЫ С priority=1:
                      В данных каждого товара указаны атрибуты в формате
                      [Атрибут=значение]. У каждого атрибута есть приоритет:
                        priority=1 — главный разделитель (включай в LABEL)
                        priority=2 — вторичный (НЕ включай в LABEL)
                        priority=3+ — игнорируй для LABEL
                      Берёт LABEL только из P1 атрибутов товара + базовый тип.

                      • Нормализуй единицы: разные написания одного значения
                        ("литров"/"л", "метров"/"м") → один формат
                      • БЕЗ маркетинговых слов и формулировок продавца
                        ("универсальный", "элитный", "для дома", "земля для")
                      • Используй ОДИН label для всех товаров с теми же P1,
                        даже если P2 (цвет, бренд, формулировка) различается

                    🚫 ЗАПРЕЩЕНО включать в LABEL значения P2 атрибутов
                      (цвет, материал, бренд, модель, второстепенные счётчики).
                      Если у двух товаров одинаковые ВСЕ P1 — это ОДИН LABEL,
                      даже если у одного цвет белый, а у другого жёлтый.

                    ✅ ПРАВИЛЬНО:
                      "25л грунт"           — для всех 25-литровых грунтов
                      "100м роса"           — для всех 100м гирлянд роса
                      "200м роса"           — для всех 200м гирлянд роса
                      "набор 142 предмета"  — для наборов на 142 предмета
                      "штангенциркуль 200мм"

                    ❌ НЕПРАВИЛЬНО (слишком детально, создаст уникальный кластер):
                      "грунт универсальный для цветов 25 литров"
                      "грунт земля для комнатных цветов универсальный 25 л"
                      "грунт, земля для цветов 25 литров"
                      "Гирлянда роса мишура светодиодная 100м"

                    Используй new:LABEL когда в кластере группа похожих товаров,
                    но они отличаются от основного кластера и нет подходящего
                    существующего кластера.

                    ⚠️ НАСЛЕДУЙ КОНТЕКСТ КЛАСТЕРА: если у большинства товаров в
                    кластере есть какой-то P1 (тип/подтип/питание), а у проверяемого
                    товара его нет в названии — считай что он унаследован из контекста
                    кластера. ★ задаёт недостающие атрибуты. Используй ОДИН label
                    для товаров с одинаковыми явными P1, не делай отдельный LABEL
                    из-за того что у одного атрибут есть в названии, а у другого нет.

                    ⚠️ ТОВАРЫ С ЧАСТИЧНЫМИ P1: если у товара извлечён ХОТЯ БЫ ОДИН
                    P1 атрибут (помечен *) и его значение НЕ совпадает с
                    доминирующим в кластере — это сигнал что товар отделяется.
                    Если в "ВСЕ товары кластера" есть ≥2 товара с тем же значением
                    этого P1 (даже если у них разные другие атрибуты или их нет) —
                    обязательно объедини их в new:LABEL по этому P1. Используй
                    короткий label из имеющегося P1: "20м", "5л", "1500Вт".
                    НЕ оставляй ok и НЕ quarantine группы ≥2 только потому что у
                    них не хватает одного из P1.
  • "quarantine"   — ТОЛЬКО для ОДИНОЧЕК. Товар уникальный, нет ни одного
                    другого товара в кластере с теми же ключевыми параметрами.

                    🚫 ЗАПРЕЩЕНО quarantine если:
                      - В секции "ВСЕ товары кластера" есть ≥1 другой товар
                        с теми же P1 параметрами (длина/объём/мощность + тип)
                      - В этом случае ОБЯЗАТЕЛЬНО new:LABEL для группы

                    ПОРОГ: 2 одинаковых товара = достаточно для new:LABEL.
                    Не жди 3+. 2 — это уже группа.

                    Пример: видишь 11 товаров "Гирлянда роса 50м" разных
                    продавцов в секции "ВСЕ товары кластера" → ВСЕМ им
                    new:50м роса. Не quarantine, не ok.

═══════════════════════════════════════════════════
ЧТО ОПРЕДЕЛЯЕТ КЛАСТЕР (ВСЕ должны совпадать)
═══════════════════════════════════════════════════

Товар принадлежит кластеру, если совпадают ВСЕ:

  ⓘ ФОРМАТ АТРИБУТОВ В ДАННЫХ:
     В скобках [...] перечислены атрибуты товара. Атрибуты с префиксом
     "*" имеют priority=1 — это ГЛАВНЫЕ РАЗДЕЛИТЕЛИ кластера. Атрибуты
     БЕЗ "*" имеют priority=2+ — они декоративные, НЕ разделяют кластер.

     Пример: [*Длина=100м, *Тип=роса, Цвет=теплый]
       *Длина и *Тип — главные (разделяют кластер)
       Цвет — вторичный (НЕ разделяет, не включай в LABEL)

  ❶ ТИП ТОВАРА — что это за вещь. ГЛАВНЫЙ РАЗДЕЛИТЕЛЬ.
     Примеры типов: контактные линзы, стремянка, набор с шуруповертом,
     раствор для линз, декоративные линзы.
     Бренд НЕ определяет тип. Acuvue и Alcon — оба "контактные линзы".
     Подтип ОПРЕДЕЛЯЕТ тип: "линзы для зрения" ≠ "декоративные линзы".
     Форм-фактор ОПРЕДЕЛЯЕТ тип: "гирлянда-лампочки" ≠ "гирлянда-бахрома".
     ⚠ Совпадение метража/фасовки НЕ делает разные типы одним кластером.
       25м лампочки ≠ 25м бахрома — тип разный, метраж не важен.

  ❷ ФАСОВКА — сколько единиц товара в упаковке
     Ключевые слова: шт, штук, линз, таблеток, капсул, пакетиков,
     мл, л, г, кг, и любые единицы измерения количества/объёма/веса.
     90шт ≠ 12шт ≠ 3шт — это РАЗНЫЕ кластеры, даже если тип одинаковый.

     ⚠ ФАСОВКА — ЖЁСТКИЙ РАЗДЕЛИТЕЛЬ. Без исключений. Без допусков.

  ❸ ЧИСЛОВЫЕ АТРИБУТЫ ИЗ СХЕМЫ — все числа со знаком и ключевые параметры
     Диоптрии (-2.50, -3.00), оптическая сила, концентрация — РАЗДЕЛИТЕЛИ.
     -2.50 ≠ -3.00 — это РАЗНЫЕ кластеры.
     Если в кластере все товары с -2.50, а проверяемый товар -3.00 → move/quarantine.
     Размер, длина, ширина — тоже разделители: 12м ≠ 20м, 1мм ≠ 1.5мм.

  ❹ КОМПЛЕКТАЦИЯ — что входит в набор/комплект
     "биты 25 шт" ≠ "биты 25 шт + насадка + кейс" — разная комплектация.
     "нейлер + 2 батареи" ≠ "нейлер + 3 батареи".
     "заклепочник вытяжной" ≠ "заклепочник резьбовой" — разный подтип.

═══════════════════════════════════════════════════
ЧТО НЕ РАЗДЕЛЯЕТ КЛАСТЕР
═══════════════════════════════════════════════════

  ✓ Радиус кривизны: 8.4, 8.6 — один кластер
  ✓ Цвет (если тот же товар): зелёный, синий — один кластер
  ✓ Размер одежды: S, M, L — один кластер
  ✓ Бренд (при одинаковом типе+фасовке): разные продавцы одного типа
  ✓ Цена: разная цена при одном типе+фасовке = ok
  ✓ Формулировка названия: один и тот же товар может называться по-разному
    у разных продавцов. "Набор отверток 25в1" = "Отвертка с набором бит 25шт"
    = "Комплект отверток 25 предметов". Если тип+фасовка+параметры совпадают → ok.
  ✓ Маркетинговые уточнения: "для суккулентов", "для дома", "профессиональный",
    "универсальный", "для начинающих" — НЕ разделители. Горшок "для суккулентов"
    = горшок "для цветов" если физически тот же товар. Разделяет ТОЛЬКО реальная
    конструктивная разница (автополив ≠ без автополива, но "для суккулентов" ≠ подтип).
  ✓ Источник питания (если базовый тип одинаковый):
    — "электрический", "электро", "электронный" = "аккумуляторный" = "беспроводной"
    — "сетевой", "проводной", "от сети 220В" — это варианты ОДНОГО типа товара
    Аккумуляторная дрель и электрическая дрель — это РАЗНЫЕ комплектации одного
    типа, но НЕ повод дробить кластер на "электрические" vs "аккумуляторные"
    подкластеры. Если в кластере перемешаны "электрические" и "аккумуляторные"
    варианты одного товара (дрель, нейлер, шуруповёрт, газонокосилка) — это
    НОРМАЛЬНО, оставлять ok. Разделять можно только если есть РАЗНЫЕ ТИПЫ
    (например бензиновый ≠ электрический — там реально другой двигатель).
  ✓ Синонимы названий измерительных приборов: "тестер" = "метр" = "анализатор"
    = "измеритель" = "детектор". pH-тестер и pH-метр — один и тот же товар.
    Не разделять кластер по тому, как продавец назвал прибор.

═══════════════════════════════════════════════════
ПРАВИЛА ДЛЯ MOVE / NEW — ПРОВЕРЯЙ ЦЕЛЕВОЙ КЛАСТЕР!
═══════════════════════════════════════════════════
⚠ ПЕРЕД решением move:K{id} — ОБЯЗАТЕЛЬНО проверь:
  1. Товары в целевом кластере (даны в контексте) — совпадают ли
     ТИП + ФАСОВКА + ПАРАМЕТРЫ + КОМПЛЕКТАЦИЯ с проверяемым товаром?
  2. Если целевой кластер тоже смешанный или неподходящий → используй new:LABEL
     или quarantine.
  3. Если нет кластера с ПОЛНЫМ совпадением → используй new:LABEL (если есть
     ≥2 похожих товаров в текущем кластере) или quarantine.
  НЕ отправляй в "ближайший похожий" — только в ТОЧНО подходящий.

⚠ КОГДА ИСПОЛЬЗОВАТЬ new:LABEL ВМЕСТО quarantine:
  Если в текущем смешанном кластере есть НЕСКОЛЬКО товаров с одинаковыми
  параметрами (например 4-5 гирлянд "роса 100м" среди разных длин), —
  все они должны получить ОДИН и тот же label "new:100м роса". Это создаст
  для них новый кластер. Quarantine — только для одиночек без пары.

═══════════════════════════════════════════════════
ХАРАКТЕРИСТИКИ ТОВАРА (не путать с фасовкой!)
═══════════════════════════════════════════════════

Характеристики описывают СВОЙСТВА товара, а не количество в упаковке:
  - "набор 118 предметов" — 118 это характеристика набора
  - "стремянка 3 ступени" — 3 это характеристика стремянки
  - "шуруповерт 18В" — 18В это характеристика

Правила для характеристик:
  • Если тип товара совпадает → характеристика ВТОРИЧНА.
    "Набор с шуруповертом 118 предметов" и "Набор с шуруповертом 95 предметов"
    → тип "набор с шуруповертом" один → ok.
  • Исключение: если кластер ЯВНО организован по характеристике
    (все товары "3 ступени"), а товар имеет другую ("6 ступеней") → move/quarantine.
  • Допуск ±15% применяется ТОЛЬКО к характеристикам, НЕ к фасовке.

КАК ОТЛИЧИТЬ ФАСОВКУ ОТ ХАРАКТЕРИСТИКИ:
  Фасовка отвечает на вопрос: "Сколько штук/единиц я получу в коробке?"
  Характеристика отвечает на вопрос: "Какие свойства у этой вещи?"
  "90 линз"    = фасовка    → жёсткий разделитель
  "118 предметов в наборе" = характеристика → мягкий, тип важнее
  "500мл бутылка"  = фасовка    → жёсткий разделитель
  "18В шуруповерт" = характеристика → мягкий

═══════════════════════════════════════════════════
ПРИОРИТЕТЫ ИСТОЧНИКОВ ИНФОРМАЦИИ
═══════════════════════════════════════════════════

1. 📷 OCR С ФОТО — если есть, доверяй больше чем названию.
   ⚠ ВНИМАНИЕ К МИНУСАМ: OCR часто теряет или искажает знак минуса
   перед числами (диоптрии, температура). Символы −, –, —, ‐ перед
   числом = минус. Если OCR показывает "2.50" а в названии "-2.50" —
   это одно и то же, минус потерян при распознавании.

2. 📝 НАЗВАНИЕ ТОВАРА — основной источник для определения типа и фасовки.

═══════════════════════════════════════════════════
ТЕГИ СИСТЕМЫ СКОРИНГА
═══════════════════════════════════════════════════
В данных товара могут быть теги от автоматического скоринга:
  [ЧИСЛО-БЛОКЕР] — фасовка или ключевой атрибут НЕ совпадает с кластером.
    → Товар ОБЯЗАТЕЛЬНО move/new/quarantine. НИКОГДА ok.
  [ЦЕНА] — цена аномальная для кластера.
    → Проверь тип+фасовку. Если совпадают → ok (цена не разделитель).
  ЧИСЛА vs кластер: ❌/✅ — автоматическое сравнение числовых атрибутов."""


def _p1_signature(product) -> tuple:
    """Build a signature from a product's P1 (priority<=1) attributes.

    Used for pre-grouping items with the same key parameters into the same
    LLM batch, so the LLM can't give inconsistent decisions to identical
    items that happen to land in different batches.

    Numeric values are bucketed using each attribute's `tolerance` field
    (from the schema). Tolerance can be:
      - relative (0 < tol < 1): bucket size = value * tol, so ±5% values
        like 82 vs 83 collapse for tol=0.05
      - absolute (tol >= 1): bucket size = tol, so ±2 items collapse for tol=2
      - None: strict equality (default for length, voltage, etc.)
    """
    if not getattr(product, "ocr_attrs", None):
        return ()
    parts = []
    for v in product.ocr_attrs.values():
        if v.get("priority", 99) > 1:
            continue
        label = v.get("label", "?")
        val = v.get("value")
        if val is None:
            continue
        if v.get("numeric"):
            try:
                fv = float(val)
                tol = v.get("tolerance")
                if tol and tol > 0:
                    if tol < 1:
                        # relative tolerance
                        bucket = max(1.0, abs(fv) * tol)
                    else:
                        bucket = float(tol)
                    norm = round(fv / bucket) * bucket
                else:
                    norm = round(fv, 2)
            except (ValueError, TypeError):
                norm = str(val).strip().lower()
        else:
            norm = str(val).strip().lower()
        parts.append((label, norm))
    # Sort by string repr to avoid TypeError when two parts share the same
    # label but have different value types (float vs str), which can happen
    # if a niche schema has duplicated labels across separators.
    return tuple(sorted(parts, key=lambda x: (str(x[0]), str(x[1]))))


# Minimum number of items in a cluster that must agree on a P1 attribute value
# before we extrapolate that value to the rest of the cluster.
INFER_FILL_MIN_SOURCES = 3


def _infer_fill_cluster_attrs(cluster_products: list) -> int:
    """Cluster-wide infer-fill for P1 attributes.

    Problem this solves: pre-grouping uses _p1_signature which counts only
    extracted attrs. If item A's OCR mentions "длина 5 м" and item B's
    OCR doesn't mention it (just "5 предметов"), they get different
    signatures and land in different batches — even though they're the
    SAME product (B's kit also has the 5m hose, just not stated on photo).

    Fix: for each P1 attribute, if N or more items in the cluster have the
    SAME extracted value AND no item has a DIFFERENT value, treat that
    value as the cluster default and inject it into items missing this
    attribute. The injected value is marked with `_inferred=True` so we
    can audit it later.

    Constraints:
      - At least INFER_FILL_MIN_SOURCES items must have the value extracted
      - All extracted values must agree (set length == 1)
      - Disagreement → real difference, never fill
      - Numeric and text P1 values both supported

    Returns the number of (item, attr) pairs that were filled in.
    """
    if len(cluster_products) < 2:
        return 0

    # attr_name -> list of (pk_id, normalized_value, source_attr_data)
    by_attr: dict[str, list] = {}
    for p in cluster_products:
        if not getattr(p, "ocr_attrs", None):
            continue
        for attr_name, attr_data in p.ocr_attrs.items():
            if attr_data.get("priority", 99) > 1:
                continue
            val = attr_data.get("value")
            if val is None:
                continue
            # Bucket numeric values to tolerance to allow ±5% as "same"
            if attr_data.get("numeric"):
                try:
                    fv = float(val)
                    tol = attr_data.get("tolerance")
                    if tol and tol > 0:
                        bucket_size = abs(fv) * tol if tol < 1 else float(tol)
                        bucket_size = max(1.0, bucket_size)
                        norm = round(fv / bucket_size) * bucket_size
                    else:
                        norm = round(fv, 2)
                except (ValueError, TypeError):
                    norm = str(val).strip().lower()
            else:
                norm = str(val).strip().lower()
            by_attr.setdefault(attr_name, []).append((p.pk_id, norm, attr_data))

    filled = 0
    for attr_name, entries in by_attr.items():
        # Must have at least N sources
        if len(entries) < INFER_FILL_MIN_SOURCES:
            continue
        # All sources must agree on the value
        unique_vals = {e[1] for e in entries}
        if len(unique_vals) != 1:
            continue

        # Use the first source's attr_data as a template (preserves label, unit, etc.)
        template = entries[0][2]
        items_with_attr = {e[0] for e in entries}

        for p in cluster_products:
            if p.pk_id in items_with_attr:
                continue
            if not p.ocr_attrs:
                p.ocr_attrs = {}
            p.ocr_attrs[attr_name] = {
                **template,
                "raw_match": "(inferred from cluster majority)",
                "_inferred": True,
            }
            filled += 1

    return filled


def _pack_batches(products: list, batch_size: int) -> list[list]:
    """Pack products into batches, keeping items with the same P1 signature
    in the same batch whenever possible.

    Strategy:
      1. Group products by their P1 signature.
      2. Sort groups by size DESC (largest first → best chance to fit whole).
      3. Greedy bin-pack: try to put each group into a batch where it fits.
      4. If a single group is larger than batch_size, split into chunks of
         batch_size — items still stay together within each chunk and the
         resulting LABEL will collapse via new_cluster_map.
    """
    by_sig: dict[tuple, list] = {}
    for p in products:
        by_sig.setdefault(_p1_signature(p), []).append(p)

    # Sort: real signatures first (so unsigned items don't crowd out groups),
    # then by group size DESC for better packing.
    groups = sorted(by_sig.items(), key=lambda kv: (kv[0] == (), -len(kv[1])))

    batches: list[list] = []
    for _sig, items in groups:
        # Items with no P1 signature: spread freely across batches as filler
        idx = 0
        while idx < len(items):
            chunk = items[idx: idx + batch_size]
            placed = False
            # Try to fit this chunk into an existing batch with room
            for b in batches:
                if len(b) + len(chunk) <= batch_size:
                    b.extend(chunk)
                    placed = True
                    break
            if not placed:
                batches.append(list(chunk))
            idx += batch_size
    return batches


def _format_attrs(ocr_attrs: dict) -> str:
    """Format ocr_attrs as a compact string for the LLM prompt.

    Marks priority=1 attributes with a leading '*' so the LLM knows which
    fields are real splitters and which are decoration. Example:
        [*Длина=100м, *Тип=роса, Цвет=теплый]
    """
    if not ocr_attrs:
        return ""
    parts = []
    for v in ocr_attrs.values():
        label = v.get("label", "?")
        raw = v.get("raw_match") or f"{v.get('value')}{v.get('unit') or ''}"
        prio = v.get("priority", 99)
        marker = "*" if prio <= 1 else ""
        parts.append(f"{marker}{label}={raw}")
    return "[" + ", ".join(parts) + "]"


def _build_prompt(
    cluster: ClusterResult,
    batch: list[Product],
    pk_to_product: dict[int, Product],
    niche_clusters: dict[int, ClusterResult] = None,
    all_cluster_products: list[Product] = None,
) -> str:
    """Build structured prompt for LLM with niche context."""
    # Cluster exemplars (top by revenue)
    exemplars = []
    sorted_pks = sorted(
        cluster.product_ids,
        key=lambda pk: pk_to_product.get(pk, Product(0, "")).revenue_1m,
        reverse=True,
    )
    for pk in sorted_pks[:5]:
        p = pk_to_product.get(pk)
        if p:
            brand_info = f" [{p.brand}]" if p.brand else ""
            price_info = f" ({p.price:.0f}₽)" if p.price > 0 else ""
            attrs_info = " " + _format_attrs(p.ocr_attrs) if p.ocr_attrs else ""
            line = f"  - {p.name}{brand_info}{price_info}{attrs_info}"
            if p.ocr_text:
                ocr_clean = " ".join(p.ocr_text.split())
                line += f"\n      📷 OCR: «{ocr_clean}»"
            exemplars.append(line)

    exemplar_text = "\n".join(exemplars) if exemplars else "  (нет примеров)"

    # Dominant attrs for current cluster
    from collections import Counter as _Counter
    dom_attrs_text = ""
    _attr_counters: dict[str, _Counter] = {}
    for pk in cluster.product_ids[:30]:
        cp = pk_to_product.get(pk)
        if cp and cp.ocr_attrs:
            for attr_name, attr_data in cp.ocr_attrs.items():
                if attr_name not in _attr_counters:
                    _attr_counters[attr_name] = _Counter()
                v = attr_data.get("value") if isinstance(attr_data, dict) else attr_data
                if v is None or isinstance(v, (list, dict, set)):
                    continue
                _attr_counters[attr_name][v] += 1
    if _attr_counters:
        dom_parts = []
        for attr_name, ctr in _attr_counters.items():
            if not ctr:
                continue
            top_val, top_cnt = ctr.most_common(1)[0]
            total = sum(ctr.values())
            if top_cnt / total >= 0.4:
                # Find label from any product
                label = attr_name[:10]
                for pk in cluster.product_ids[:5]:
                    cp = pk_to_product.get(pk)
                    if cp and cp.ocr_attrs and attr_name in cp.ocr_attrs:
                        label = cp.ocr_attrs[attr_name].get("label", label)
                        break
                dom_parts.append(f"{label}={top_val} ({top_cnt}/{total})")
        if dom_parts:
            dom_attrs_text = "\n  Доминантные атрибуты кластера: " + ", ".join(dom_parts)

    # Other clusters in niche (for move targets)
    other_clusters_text = ""
    if niche_clusters:
        other = []
        for gid, cl in sorted(niche_clusters.items(), key=lambda x: -x[1].size)[:20]:
            if gid == cluster.gid:
                continue
            # Get first product name as label
            first_pk = cl.product_ids[0] if cl.product_ids else None
            first_p = pk_to_product.get(first_pk) if first_pk else None
            label = first_p.name[:60] if first_p else f"cluster_{gid}"
            price_info = f", ~{cl.avg_price:.0f}₽" if cl.avg_price > 0 else ""
            # Show attrs if available
            attrs_info = " " + _format_attrs(first_p.ocr_attrs) if (first_p and first_p.ocr_attrs) else ""
            other.append(f"  K{gid}: \"{label}\"{attrs_info} ({cl.size} товаров{price_info})")
        if other:
            other_clusters_text = "\n\nДругие кластеры ниши (для move):\n" + "\n".join(other)

    # Products to check
    # Pre-compute number verdicts for each product vs cluster
    def _num_verdict(p_product):
        if not p_product.ocr_attrs or not _attr_counters:
            return ""
        verdicts = []
        for attr_name, ctr in _attr_counters.items():
            if attr_name not in p_product.ocr_attrs:
                continue
            pa = p_product.ocr_attrs[attr_name]
            if not pa.get("numeric"):
                continue
            p_val = pa.get("value")
            if p_val is None:
                continue
            # Find dominant group (±15%)
            vals = sorted(_safe_float(v) for v in ctr.elements() if v is not None and _safe_float(v) is not None)
            if not vals:
                continue
            best_group = []
            for anchor in vals:
                group = [v for v in vals if anchor * 0.85 <= v <= anchor * 1.15]
                if len(group) > len(best_group):
                    best_group = group
            if not best_group or len(best_group) / len(vals) < 0.4:
                continue
            median = sorted(best_group)[len(best_group) // 2]
            try:
                pf = float(p_val)
                ratio = max(pf, median) / min(pf, median) if pf > 0 and median > 0 else 999
                label = pa.get("label", attr_name[:6])
                if ratio <= 1.15:
                    verdicts.append(f"✅{label}={p_val}≈кластер({median})")
                else:
                    verdicts.append(f"❌{label}={p_val}≠кластер({median})")
            except (ValueError, TypeError):
                pass
        return " ".join(verdicts)

    items = []
    for p in batch:
        head_parts = [f'pk_id={p.pk_id}: "{p.name}"']
        if p.brand:
            head_parts.append(f"бренд={p.brand}")
        if p.price > 0:
            head_parts.append(f"цена={p.price:.0f}₽")
        head_parts.append(f"score={p.cumulative_score:.2f}")

        sub_lines = []
        if p.ocr_attrs:
            sub_lines.append(f"      атрибуты: {_format_attrs(p.ocr_attrs)}")
        verdict = _num_verdict(p)
        if verdict:
            sub_lines.append(f"      ЧИСЛА vs кластер: {verdict}")
        if p.ocr_text:
            ocr_clean = " ".join(p.ocr_text.split())  # collapse whitespace
            sub_lines.append(f"      📷 OCR с фото карточки:\n        «{ocr_clean}»")
        else:
            sub_lines.append("      📷 OCR с фото карточки: (нет)")

        items.append("  " + ", ".join(head_parts) + "\n" + "\n".join(sub_lines))

    items_text = "\n".join(items)

    # ── ALL items in cluster (so LLM can spot subgroups across batches) ──
    # Compact one-line view of every product in this cluster, ordered by score.
    # This lets the LLM use new:LABEL with consistent labels across batches:
    # if it sees 5 items "Гирлянда роса 50м" in the global view, it will use
    # the same `new:50м роса` label even if only 1-2 of them are in the
    # current batch.
    all_items_text = ""
    if all_cluster_products:
        sorted_all = sorted(
            all_cluster_products,
            key=lambda x: -x.cumulative_score,
        )
        all_lines = []
        for p in sorted_all:
            attrs_short = " " + _format_attrs(p.ocr_attrs) if p.ocr_attrs else ""
            price_short = f" {p.price:.0f}₽" if p.price > 0 else ""
            all_lines.append(f"  pk={p.pk_id}: {p.name}{price_short}{attrs_short}")
        all_items_text = (
            "\n\nВСЕ товары текущего кластера (для согласованных new:LABEL "
            "между батчами — используй одинаковый LABEL для одинаковых "
            "подгрупп):\n" + "\n".join(all_lines)
        )

    return f"""Текущий кластер K{cluster.gid} (типичные товары):
{exemplar_text}{dom_attrs_text}
{other_clusters_text}{all_items_text}

Товары для проверки (ok / move:K{{ID}} / new:LABEL / quarantine):
{items_text}

Ответ СТРОГО JSON массивом."""


async def _call_llm(prompt: str) -> list[dict]:
    """Call LLM and parse response."""
    if not OPENROUTER_API_KEY:
        log.warning("No OPENROUTER_API_KEY, skipping LLM")
        return []

    try:
        async with httpx.AsyncClient(timeout=120.0) as client:
            r = await client.post(
                f"{OPENROUTER_BASE}/chat/completions",
                headers={
                    "Authorization": f"Bearer {OPENROUTER_API_KEY}",
                    "Content-Type": "application/json",
                },
                json={
                    "model": LLM_MODEL,
                    "messages": [
                        {"role": "system", "content": SYSTEM_PROMPT},
                        {"role": "user", "content": prompt},
                    ],
                    "max_tokens": 4000,
                    "temperature": 0.1,
                },
            )

        if r.status_code != 200:
            log.warning(f"LLM API error {r.status_code}: {r.text[:200]}")
            return []

        data = r.json()
        txt = data["choices"][0]["message"]["content"]

        # Extract JSON array
        a = txt.find("[")
        b = txt.rfind("]") + 1
        if a < 0 or b <= a:
            log.warning(f"LLM response has no JSON array: {txt[:200]}")
            return []

        return json.loads(txt[a:b])

    except Exception as e:
        log.warning(f"LLM call failed: {e}")
        return []


def _p1_cluster_median(cluster: ClusterResult, attr_name: str, pk_to_product: dict) -> float | None:
    from collections import Counter
    counter: Counter = Counter()
    for pk in cluster.product_ids[:30]:
        p = pk_to_product.get(pk)
        if not (p and p.ocr_attrs):
            continue
        ad = p.ocr_attrs.get(attr_name)
        if not isinstance(ad, dict):
            continue
        val = ad.get("value")
        if val is None or isinstance(val, (list, dict, set)):
            continue
        counter[val] += 1
    vals = [f for v in counter.elements() if (f := _safe_float(v)) is not None]
    if not vals:
        return None
    vals_sorted = sorted(vals)
    best_group: list[float] = []
    for anchor in vals_sorted:
        group = [v for v in vals_sorted if anchor * 0.85 <= v <= anchor * 1.15]
        if len(group) > len(best_group):
            best_group = group
    if len(best_group) / len(vals) < 0.4:
        return None
    return sorted(best_group)[len(best_group) // 2]


TARGET_HARD_CAP_RATIO = 2.0
TARGET_SOFT_CAP_RATIO = 1.15


def _validate_move_p1(
    product: Product,
    target_cluster: ClusterResult,
    pk_to_product: dict,
    source_cluster: ClusterResult | None = None,
) -> bool:
    if not product.ocr_attrs:
        return True

    for attr_name, p_attr in product.ocr_attrs.items():
        if not isinstance(p_attr, dict):
            continue
        if p_attr.get("priority", 99) > 1 or not p_attr.get("numeric", False):
            continue
        p_val = p_attr.get("value")
        if p_val is None:
            continue
        try:
            pf = float(p_val)
        except (ValueError, TypeError):
            continue
        if pf <= 0:
            continue

        tgt_median = _p1_cluster_median(target_cluster, attr_name, pk_to_product)
        if tgt_median is None or tgt_median <= 0:
            continue

        tgt_ratio = max(pf, tgt_median) / min(pf, tgt_median)
        if tgt_ratio <= TARGET_SOFT_CAP_RATIO:
            continue

        if tgt_ratio > TARGET_HARD_CAP_RATIO:
            log.info(
                f"  P1 REJECT move pk={product.pk_id}: {attr_name} product={pf}"
                f" target={tgt_median} ratio={tgt_ratio:.2f} > hard cap {TARGET_HARD_CAP_RATIO}"
            )
            return False

        src_median = (
            _p1_cluster_median(source_cluster, attr_name, pk_to_product)
            if source_cluster is not None else None
        )
        if src_median is None or src_median <= 0:
            log.info(
                f"  P1 REJECT move pk={product.pk_id}: {attr_name} product={pf}"
                f" target={tgt_median} ratio={tgt_ratio:.2f} (no source baseline)"
            )
            return False

        src_ratio = max(pf, src_median) / min(pf, src_median)
        if tgt_ratio >= src_ratio:
            log.info(
                f"  P1 REJECT move pk={product.pk_id}: {attr_name} product={pf}"
                f" source={src_median} (r={src_ratio:.2f}) vs target={tgt_median} (r={tgt_ratio:.2f})"
                f" — target not better"
            )
            return False

        log.info(
            f"  P1 RELATIVE-OK pk={product.pk_id}: {attr_name} product={pf}"
            f" source={src_median} (r={src_ratio:.2f}) → target={tgt_median} (r={tgt_ratio:.2f})"
        )
    return True


async def arbitrate_grey_zone(
    products: list[Product],
    clusters: dict[int, ClusterResult],
) -> dict[str, int]:
    """
    LLM arbitration for products with decision == 'grey'.
    Updates product.decision and product.confidence in place.
    """
    grey = [p for p in products if p.decision == "grey"]
    if not grey:
        log.info("LLM: No grey products to arbitrate")
        return {"llm_ok": 0, "llm_quarantine": 0, "llm_move": 0,
                "llm_split": 0, "llm_failed": 0}

    pk_to_product = {p.pk_id: p for p in products}

    # Allocate fresh cluster_gids for "new:LABEL" decisions.
    # We start above the highest existing gid and increment per (source, label) pair.
    try:
        from .db import get_engine
        from sqlalchemy import text as _sa_text
        with get_engine().connect() as _conn:
            _max_gid = _conn.execute(_sa_text(
                "SELECT COALESCE(MAX(cluster_gid),0) FROM mpstats_product_clusters"
            )).scalar()
        new_gid_counter = [int(_max_gid) + 1]
    except Exception as _e:
        log.warning(f"Failed to query max cluster_gid, starting from 800_000_000: {_e}")
        new_gid_counter = [800_000_000]
    # Map (source_cluster_gid, label_str) → fresh allocated gid (so same label
    # across batches in the same cluster collapses into one new cluster).
    new_cluster_map: dict[tuple[int, str], int] = {}
    # Track every split assignment so we can post-process singletons:
    #   list of (product, source_gid, new_gid, label)
    splits_made: list = []

    # If a cluster has any grey product, send ALL its products through LLM.
    # Reason: scoring sometimes leaves "ok" twins of clearly-broken products
    # behind (one product slipped grey, its twin slipped ok). LLM should see
    # the full cluster and re-decide consistently for everyone.
    clusters_with_grey = {p.new_cluster_gid for p in grey}

    by_cluster: dict[int, list[Product]] = {}
    for p in products:
        if p.new_cluster_gid in clusters_with_grey:
            by_cluster.setdefault(p.new_cluster_gid, []).append(p)

    total_to_check = sum(len(v) for v in by_cluster.values())
    log.info(
        f"LLM: Arbitrating {len(grey)} grey products + {total_to_check - len(grey)} "
        f"twins from {len(clusters_with_grey)} affected clusters (total {total_to_check})"
    )

    # Build niche → clusters map for move targets
    niche_to_clusters: dict[int, dict[int, ClusterResult]] = {}
    for gid, cl in clusters.items():
        # Find niche from first product
        first_p = pk_to_product.get(cl.product_ids[0]) if cl.product_ids else None
        if first_p:
            niche_to_clusters.setdefault(first_p.niche_key, {})[gid] = cl

    stats = {"llm_ok": 0, "llm_quarantine": 0, "llm_move": 0,
             "llm_split": 0, "llm_failed": 0}

    for gid, cluster_products in by_cluster.items():
        cluster = clusters.get(gid)
        if not cluster:
            stats["llm_failed"] += len(cluster_products)
            continue

        # Get niche clusters for move context
        first_p = pk_to_product.get(cluster.product_ids[0]) if cluster.product_ids else None
        niche_cls = niche_to_clusters.get(first_p.niche_key, {}) if first_p else {}

        # Cluster-wide infer-fill: if N+ items in this cluster agree on a P1
        # attribute value, propagate it to items missing the attr. Prevents
        # false-splits caused by uneven OCR coverage of the same parameter.
        filled = _infer_fill_cluster_attrs(cluster_products)
        if filled:
            log.info(
                f"  K{gid}: infer-fill propagated {filled} P1 values "
                f"from cluster majority"
            )

        # Process in batches. Pre-group by P1 signature so items with the
        # same key parameters land in the SAME LLM batch — this prevents the
        # LLM from giving inconsistent decisions to identical products that
        # would otherwise be split across batches.
        # Also pass `cluster_products` (full list) so the prompt's global
        # view shows all items, helping the LLM pick consistent new:LABEL.
        packed = _pack_batches(cluster_products, LLM_BATCH_SIZE)
        log.info(
            f"  K{gid}: packed {len(cluster_products)} items into {len(packed)} "
            f"signature-aware batches"
        )
        for batch in packed:
            prompt = _build_prompt(
                cluster, batch, pk_to_product, niche_cls,
                all_cluster_products=cluster_products,
            )
            results = await _call_llm(prompt)

            # Map results
            result_map = {}
            for r in results:
                pk = r.get("pk_id")
                if pk:
                    result_map[int(pk)] = r

            for p in batch:
                r = result_map.get(p.pk_id)
                if not r:
                    p.confidence = 0.5
                    stats["llm_failed"] += 1
                    continue

                decision = str(r.get("decision", "")).strip()
                p.decided_by = "llm"
                p.reason = r.get("reason", "")
                p.confidence = float(r.get("confidence", 0.5))

                if decision == "ok":
                    p.decision = "ok"
                    stats["llm_ok"] += 1
                elif decision.startswith("move:"):
                    # Parse target cluster: "move:K50790" or "move:50790"
                    try:
                        # Self-contradiction guard: LLM mentions a difference
                        # in its own reason but still decides to move → quarantine for review.
                        reason_lower = (r.get("reason") or "").lower()
                        if _reason_contradicts_move(reason_lower):
                            p.decision = "quarantine"
                            p.reason += " | [ПЕРЕПРОВЕРКА] LLM упомянул различие в reason"
                            stats["llm_quarantine"] += 1
                            log.info(f"  RECHECK pk={p.pk_id}: reason={reason_lower[:80]!r} → quarantine for review")
                            continue

                        target_gid = int(decision.split(":")[1].lstrip("Kk"))
                        if target_gid in clusters and target_gid != p.new_cluster_gid:
                            target_cl = clusters[target_gid]
                            # Niche guard: target must be in same niche as source.
                            src_niche = p.niche_key
                            tgt_first = pk_to_product.get(target_cl.product_ids[0]) if target_cl.product_ids else None
                            tgt_niche = tgt_first.niche_key if tgt_first else None

                            # Text-overlap guard: source name must share ≥1
                            # meaningful (non-digit, non-stopword) token with
                            # target main product name. Blocks LLM hallucinating
                            # random K-id where target is a totally different
                            # product type (e.g. moving "набор пневмоинструмента
                            # 71 предмет" to "шланг резиновый 15 метров").
                            tgt_main_p = (
                                pk_to_product.get(target_cl.main_pk)
                                if target_cl.main_pk else None
                            )
                            src_tokens = {
                                t for t in tokenize(p.name or "")
                                if not t.isdigit()
                            }
                            tgt_tokens = {
                                t for t in tokenize(tgt_main_p.name or "")
                                if not t.isdigit()
                            } if tgt_main_p else set()
                            text_overlap = bool(src_tokens & tgt_tokens)

                            if src_niche and tgt_niche and src_niche != tgt_niche:
                                p.decision = "quarantine"
                                p.reason += f" | target K{target_gid} в чужой нише ({tgt_niche} vs {src_niche}) → quarantine"
                                stats["llm_quarantine"] += 1
                                log.info(f"  NICHE REJECT pk={p.pk_id}: target K{target_gid} niche={tgt_niche} != src niche={src_niche}")
                            elif not text_overlap and tgt_main_p:
                                p.decision = "quarantine"
                                p.reason += (
                                    f" | target K{target_gid} text-mismatch "
                                    f"(no shared tokens with main {tgt_main_p.name[:40]!r}) → quarantine"
                                )
                                stats["llm_quarantine"] += 1
                                log.info(
                                    f"  TEXT REJECT pk={p.pk_id}: target K{target_gid} "
                                    f"main={tgt_main_p.name[:60]!r}"
                                )
                            elif _validate_move_p1(
                                p, target_cl, pk_to_product,
                                source_cluster=clusters.get(p.new_cluster_gid),
                            ):
                                p.decision = "move"
                                p.new_cluster_gid = target_gid
                                stats["llm_move"] += 1
                            else:
                                p.decision = "quarantine"
                                p.reason += " | P1 conflict с target → quarantine"
                                stats["llm_quarantine"] += 1
                        else:
                            p.decision = "quarantine"
                            stats["llm_quarantine"] += 1
                    except (ValueError, IndexError):
                        p.decision = "quarantine"
                        stats["llm_quarantine"] += 1
                elif decision.startswith("new:"):
                    # Split decision: collect items with the same LABEL into a new cluster.
                    label = decision.split(":", 1)[1].strip().lower()
                    if not label:
                        p.decision = "quarantine"
                        p.reason += " | empty new: label → quarantine"
                        stats["llm_quarantine"] += 1
                    else:
                        key = (gid, label)
                        if key not in new_cluster_map:
                            new_cluster_map[key] = new_gid_counter[0]
                            new_gid_counter[0] += 1
                            log.info(
                                f"  NEW CLUSTER allocated gid={new_cluster_map[key]} "
                                f"from source K{gid} label={label!r}"
                            )
                        p.decision = "move"
                        p.new_cluster_gid = new_cluster_map[key]
                        p.reason = f"new:{label} | {p.reason}".strip(" |")
                        stats["llm_split"] += 1
                        splits_made.append((p, gid, new_cluster_map[key], label))
                elif decision == "quarantine":
                    p.decision = "quarantine"
                    stats["llm_quarantine"] += 1
                else:
                    p.confidence = 0.5
                    stats["llm_failed"] += 1

    # ── Post-process: revert singleton splits without niche-wide pair ─────
    # If LLM created a new:LABEL with only 1 product, check whether ANY other
    # product in the same niche (across all clusters) has the same P1 signature.
    # If yes — keep the split (it may merge with the other cluster later).
    # If no — there's no pair anywhere, so creating a new singleton cluster is
    # noise; revert to "ok" in the original cluster.
    # Hard rule: a new:LABEL split is only kept if it has at least
    # NEW_CLUSTER_MIN_SIZE items. Smaller groups (1-4 items) ALWAYS revert
    # to source — too risky to spawn micro-clusters from LLM noise.
    NEW_CLUSTER_MIN_SIZE = 4
    if splits_made:
        # Group splits by new_gid
        by_new_gid: dict[int, list] = {}
        for entry in splits_made:
            by_new_gid.setdefault(entry[2], []).append(entry)

        reverted = 0
        for new_gid, entries in by_new_gid.items():
            if not entries:
                continue
            if len(entries) >= NEW_CLUSTER_MIN_SIZE:
                continue  # big enough → keep as new sub-cluster
            label = entries[0][3]
            for p, source_gid, _, lbl in entries:
                p.decision = "ok"
                p.new_cluster_gid = source_gid
                p.reason = (f"tiny-revert (only {len(entries)} items, "
                            f"need ≥{NEW_CLUSTER_MIN_SIZE}) | was new:{lbl}")
                stats["llm_split"] -= 1
                stats["llm_ok"] += 1
                reverted += 1
            key_to_drop = (entries[0][1], label)
            if key_to_drop in new_cluster_map:
                del new_cluster_map[key_to_drop]

        if reverted:
            log.info(
                f"  Reverted {reverted} small new-cluster items "
                f"(< {NEW_CLUSTER_MIN_SIZE} items each)"
            )

    log.info(
        f"LLM results: ok={stats['llm_ok']}, "
        f"move={stats['llm_move']}, "
        f"split={stats['llm_split']} ({len(new_cluster_map)} new clusters), "
        f"quarantine={stats['llm_quarantine']}, "
        f"failed={stats['llm_failed']}"
    )
    return stats
