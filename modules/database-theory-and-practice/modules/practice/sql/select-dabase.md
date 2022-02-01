# select database()

## Conteúdo

 - [01 - Introdução ao comando "select database()"](#intro)

---

<div id="intro"></div>

## 01 - Introdução ao comando "select database()"

> O comando **select database();** é responsável por dizer em qual Banco de Dados estamos conectados.

Por exemplo:

```sql
select database();
```

**OUTPUT:**  
```sql
+------------+
| database() |
+------------+
| NULL       |
+------------+
1 row in set (0.00 sec)
```

**NOTE:**  
> Como nós não estamos conectados a nenhum Banco de Dados o resultado foi **NULL**.

Agora vamos nos conectar a um Banco de Dados qualquer para ver como fica:

```sql
mysql> use bd_library
Database changed
```

```sql
select database();
```

**OUTPUT:**  
```sql
+------------+
| database() |
+------------+
| bd_library |
+------------+
1 row in set (0.00 sec)
```

**NOTE:**  
Veja que agora realmente nós estamos conectados em um Banco de Dados: **bd_library**.

---

**Rodrigo Leite -** *drigols*