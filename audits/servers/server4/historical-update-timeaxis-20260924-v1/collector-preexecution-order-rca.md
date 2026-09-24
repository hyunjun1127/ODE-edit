# Collector 저장 순서 최소 수리

실제 GPU/CPU job 실패가 아니라 제출 후 CPU source 실패경로 검토에서 발견했다. 실행 source6ef71ed2의 `reduce.main`은 T3A/table/figure를 만든 뒤, report-ko.md·artifact-index.json보다 먼저 COMPLETED terminal을 생성했다. 마지막 report/inventory I/O exception이면 terminal과 collector exit가 불일치할 수 있었다. 해당 실제 exception은 아직 발생하지 않았다.

수리: report와 inventory 성공·SHA 확보 뒤 마지막 create-once write로 COMPLETED를 기록한다. 계산식·수치기준·panel·GPU worker·원 execution lock은 변경0. 좁은 CPU regression으로 저장 순서를 검사하고, 아직 pending인 원 CPU collector53178만 exact owner/source/afterany를 확인하여 교체한다. GPU53176/53177은 유지한다. 원 source/archive/submission 보존, 원 collector는 실행 전 취소 및 allocation0을 실제 accounting으로 구분한다. 새 collector source/ID는 별도 receipt다.

권한: 동일 task의 정의 변경 없는 기술 오류 최소수리·새 source/attempt·영향 downstream 재등록 허용. 별도 재승인 대기 없음. 과학 음성이나 GPU numerical 결과에 따른 변경이 아니다.
