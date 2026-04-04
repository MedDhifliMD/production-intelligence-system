import csv
import ijson
import re
import os

def build_verification_dict(filepath):
    v_dict = {}
    doc_count = 0
    with open(filepath, 'rb') as f:
        for doc in ijson.items(f, 'item'):
            doc_count += 1
            if doc_count % 10000 == 0:
                print(f"Loaded {doc_count} verification records into memory...", flush=True)
                
            executions = doc.get('Executions', [])
            if not executions: continue
            
            exec_0 = executions[0]
            bv = exec_0.get('boardVerification', {})
            
            timestamp = None
            ts_obj = bv.get('TimeStamp')
            if isinstance(ts_obj, dict):
                timestamp = ts_obj.get('$date')
            elif isinstance(ts_obj, str):
                timestamp = ts_obj
                
            body = bv.get('MessageBody', {})
            for unit in body.get('InspectedUnits', []):
                pattern_barcode = unit.get('UnitIdentifier')
                if not pattern_barcode:
                    continue
                
                if pattern_barcode not in v_dict:
                    v_dict[pattern_barcode] = {
                        'Verif_Date': timestamp,
                        'defects': {}
                    }
                
                for insp in unit.get('Inspections', []):
                    for defct in insp.get('DefectsFound', []):
                        raw_desig = defct.get('ComponentOfInterest', {}).get('ReferenceDesignator', '')
                        if raw_desig:
                            clean_desig = re.split(r'[-_\s]', raw_desig)[0]
                            v_dict[pattern_barcode]['defects'][clean_desig] = {
                                'Componenet_Result': defct.get('ComponenetResult', 'Fail'),
                                'DefectCode': defct.get('DefectCode', '')
                            }
    print(f"Verification Load Complete! Unique Boards tracked: {len(v_dict)}")
    return v_dict

def stream_npm_to_csv(npm_filepath, v_dict, out_csv):
    headers = [
        "Barcode", "NPM_Date", "Verif_Date", "Pattern_Barcode", "Pattern_Index", 
        "Designator", "Componenet_Result", "Feede_ID", "Nozel_Name", "Coordinate_X", 
        "Coordinate_Y", "Rotation", "Has_Verification", "Has_Component_Verification", "DefectCode"
    ]
    
    components_written = 0
    doc_count = 0
    with open(out_csv, 'w', newline='', encoding='utf-8') as cout:
        writer = csv.writer(cout)
        writer.writerow(headers)
        
        with open(npm_filepath, 'rb') as f:
            for doc in ijson.items(f, 'item'):
                doc_count += 1
                if doc_count % 5000 == 0:
                    print(f"Streamed {doc_count} NPM Boards. Components converted: {components_written}...", flush=True)
                
                executions = doc.get('Executions', [])
                if not executions: continue
                exec_0 = executions[0]
                
                board = exec_0.get('BoardNpm_VF', {})
                barcode = board.get('barcode', '')
                npm_date = board.get('lastProcessingTime', '')
                
                mat_lookup = {m.get('materialReferenceLink'): m.get('feederID') for m in board.get('materials', [])}
                nzl_lookup = {nz.get('nzlReferenceLink'): nz.get('name') for nz in board.get('nzls', [])}
                
                for pattern in board.get('patterns', []):
                    pat_barcode = pattern.get('patternBarcode', '')
                    pat_index = pattern.get('index', '')
                    
                    v_info = v_dict.get(pat_barcode)
                    has_verif = (v_info is not None)
                    verif_date = v_info['Verif_Date'] if has_verif else ''
                    
                    for comp in pattern.get('components', []):
                        desig = comp.get('Designator', '')
                        comp_res = "Pass"
                        has_comp_verif = False
                        defect_code = ""
                        
                        if has_verif:
                            comp_defects = v_info['defects']
                            if desig in comp_defects:
                                has_comp_verif = True
                                comp_res = comp_defects[desig].get('Componenet_Result', 'Pass')
                                defect_code = comp_defects[desig].get('DefectCode', '')
                                
                        mat_link = comp.get('materialReferenceLink')
                        feeder_id = mat_lookup.get(mat_link, '')
                        
                        nzl_link = comp.get('nzlReferenceLink')
                        nozzle = nzl_lookup.get(nzl_link, '')
                        
                        coords = comp.get('coordinate', {})
                        x = coords.get('coordinateX', '')
                        y = coords.get('coordinateY', '')
                        rot = coords.get('Rotation', '')
                        
                        writer.writerow([
                            barcode, npm_date, verif_date, pat_barcode, pat_index,
                            desig, comp_res, feeder_id, nozzle, x, y, rot,
                            has_verif, has_comp_verif, defect_code
                        ])
                        components_written += 1
                        
    return components_written

def main():
    # Directly targeting the huge 2.5 GB database files!
    verification_path = r"C:\Users\dhifl\Desktop\ai training\Verification-Station.json"
    npm_path = r"C:\Users\dhifl\Desktop\ai training\Ai-Dataset.Npm.json"

    out_csv = "final_dataset.csv"
    
    print(f"Starting pipeline. Target CSV: {out_csv}")
    v_dict = build_verification_dict(verification_path)
    count = stream_npm_to_csv(npm_path, v_dict, out_csv)
    print(f"PIPELINE COMPLETE! Giant CSV successfully generated with {count} component rows!")

if __name__ == "__main__":
    main()
