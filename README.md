# carom_video2data_01 (v001)

당구 경기 영상(SOOP 등)을 컴퓨터 비전(YOLO/Tracking)과 물리 엔진을 통해 분석하여 타격 데이터(당점, 속도, 두께) 및 성공/실패 여부를 추출하는 자동화 파이프라인.

## Features
1. **Data Acquisition**: 경기 VOD 다운로드 및 메타데이터 추출
2. **Segmentation**: 점수판 OCR을 통한 이닝(Round) 분할 및 샷 단위 클립 생성
3. **Vision & Physics Engine**: 2D 좌표 변환 및 물리 역운동학 계산
4. **Database**: 분석된 계층형 데이터(Match -> Inning -> Shot) 관리 (SQLite)
