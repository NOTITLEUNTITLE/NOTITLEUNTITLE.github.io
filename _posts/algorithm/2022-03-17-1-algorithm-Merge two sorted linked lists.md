---
layout: single
title:  "[Hackerrank] Merge two sorted linked lists"
categories: algorithm
tag: [python, algorithm]
toc: true
author_profile: false
---


# 문제 설명
Given pointers to the heads of two sorted linked lists, merge them into a single, sorted linked list. Either head pointer may be null meaning that the corresponding list is empty.
<br/><br/>
두개의 정렬된 리스트들의 head를 가지고서, 하나의 정렬된 리스트의 head를 반환하는 문제입니다!!


<br/><br/><br/>

# 제한사항

## Input Format

- 2개의 head

## Output Format

- SinglyLinkedListNode pointer: a reference to the head of the merged list<br/>
(SinglyLinkedListNode의 head를 넘겨주면 됩니다.)


## Others
- 중요하지 않아 보입니다!

<br/><br/><br/>


# 입출력 예

- 생략

# Idea

<p>
사실 문제를 보고 쉽게 풀었습니다.<br/>
하지만 다른사람의 풀이를 보고서, 공부하고 기록할겸 풀이를 남깁니다.<br/>
어렵지는 않습니다!!<br/>

</p>
<br/><br/><br/>

# Code

```python
def mergeLists(head1, head2):
    sorted_list = SinglyLinkedList()
    shit = list()
    while head1:
        # sorted_list.insert_node(head1.data)
        shit.append(head1.data)
        # print(head1.data, end=' ')
        head1 = head1.next
    # print()
    while head2:
        # sorted_list.insert_node(head2.data)
        shit.append(head2.data)
        # print(head2.data, end=' ')
        head2 = head2.next
    shit.sort()
    for sh in shit:
        sorted_list.insert_node(sh)
    test = sorted_list.head
    while test:
        print(test.data, end=' ')
        test = test.next
    
    return sorted_list.head
```


# Explain
각 head별로 순회(?)하면서 data들만 list에 따로 넣어줍니다.<br/>
그리고 리스트를 정렬한 후,<br/>
객체의 head를 return해야 하기에 객체에 순서대로 넣어줍니다.<br/>


<br/><br/><br/>


## 다른사람의 풀이

```python
def mergeLists(head1, head2):
    if not head1 and not head2:
        return head1
    if not head1:
        return head2
    if not head2:
        return head1
    merged_list = SinglyLinkedListNode(None)
    c = merged_list
    c1 = head1
    c2 = head2
    while c1 and c2:
        print(merged_list)
        if c1.data <= c2.data:
            c.next = c1
            c1 = c1.next
        else:
            c.next = c2
            c2 = c2.next
        c = c.next
    if c1:
        c.next = c1
    if c2:
        c.next = c2
    return merged_list.next
```
감히 평가하자면 파이썬스럽지 않은 코드이지만, 그럼에도 잘 짯습니다.
위의 코드를 조금 수정하자면 아래와 같이 나타낼 수 있을것 같습니다.
```python
def mergeLists(head1, head2):
    merged_list = SinglyLinkedList()
    
    while head1 or head2:
        if not head1:
            merged_list.insert_node(head2.data)
            head2 = head2.next
        elif not head2:
            merged_list.insert_node(head1.data)
            head1 = head1.next
        elif head1.data < head2.data:
            merged_list.insert_node(head1.data)
            head1 = head1.next
        else:
            merged_list.insert_node(head2.data)
            head2 = head2.next
    return merged_list.head

```


<!-- # References

<ul>
  <li><a href="https://medium.com/@mrunankmistry52/non-divisible-subset-problem-comprehensive-explanation-c878a752f057" target="_blank">https://medium.com/@mrunankmistry52/non-divisible-subset-problem-comprehensive-explanation-c878a752f057</a></li>
  <li><a href="https://gaegosoo.tistory.com/62" target="_blank">https://gaegosoo.tistory.com/62</a></li>
  
</ul> -->

